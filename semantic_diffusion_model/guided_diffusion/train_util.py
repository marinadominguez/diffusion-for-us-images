# this script has been modified to incorporate w&b 
# 07/11/2023
from typing import List, Dict

import copy
import functools
import os
os.environ["MPLBACKEND"] = "Agg"

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

import gc
import torch
#import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

import json, shutil
from dataclasses import dataclass

import os
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from config import cfg

try:
    import wandb
except Exception:
    wandb = None

def is_rank0():
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
            self,
            *,
            model,
            diffusion,
            data,
            num_classes,
            batch_size,
            microbatch,
            lr,
            ema_rate,
            drop_rate,
            log_interval,
            save_interval,
            resume_checkpoint,
            use_fp16=False,
            fp16_scale_growth = 1e-3,
            schedule_sampler=None,
            weight_decay=1e-3,
            lr_anneal_steps=0,
            output_dir,
            distributed_data_parallel=False,
            val_data=None,
            val_interval=200,
            val_batches=10,

    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size

        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.drop_rate = drop_rate
        self.log_interval = log_interval 
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.output_dir = output_dir 
        self.distributed_data_parallel = distributed_data_parallel

        self.step = 0
        self.resume_step = 0
        #self.global_batch = self.batch_size * dist.get_world_size()
        self.global_batch = self.batch_size * (dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1)
        self.world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        
        self.last_train_loss = None
        self.val_data = val_data
        self.val_interval = val_interval
        self.val_batches = val_batches

        self.best_json_path = os.path.join(self.output_dir, "best.json")
        self.best_dir = os.path.join(self.output_dir, "best")
        self.best_eval_dir = os.path.join(self.output_dir, "best_eval")
        self._lpips = None
        self._ssim = None

        self._real_pool = None        # tensor (N,1,H,W) o (N,3,H,W)
        self._cond_pool = None        # condicionamiento asociado (para muestrear condicionado)

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        """if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model
        
        # Clear CUDA cache before starting the training
        torch.cuda.empty_cache()"""

        ddp_ok = (
            self.distributed_data_parallel
            and th.cuda.is_available()
            and dist.is_available()
            and dist.is_initialized()
            and (self.world_size > 1)
        )

        if ddp_ok:
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            self.use_ddp = False
            self.ddp_model = self.model
    
    def _get_loss_simple_key(self, losses: Dict[str, th.Tensor]) -> str:
        # guided-diffusion suele llamar "mse" al término simple loss
        return "mse" if "mse" in losses else ("loss" if "loss" in losses else list(losses.keys())[0])
    
    @th.no_grad()
    def _nn_match_metrics(self, gen_x: th.Tensor, real_pool: th.Tensor, lpips_fn, ssim_fn, chunk_real: int = 64):
        """
        gen_x: (B,3,H,W) [-1,1]
        real_pool: (N,1,H,W) [-1,1]   (guardado en 1ch)
        return: mean_ssim_nn, mean_lpips_nn
        """
        B = gen_x.shape[0]
        gen_rgb = _as_lpips_rgb(gen_x).to(dist_util.dev())
        gen_gray = _as_gray_1ch(gen_x).to(dist_util.dev())

        real_gray = real_pool.to(dist_util.dev())
        real_rgb = _as_lpips_rgb(real_gray)  # 1ch->3ch

        best_lp = th.full((B,), float("inf"), device=dist_util.dev())
        best_idx = th.zeros((B,), dtype=th.long, device=dist_util.dev())

        # barrido por chunks del pool
        N = real_rgb.shape[0]
        for j in range(0, N, chunk_real):
            rr = real_rgb[j:j+chunk_real]               # (C,3,H,W)
            # LPIPS por pares: repetimos gen vs cada real
            # forma fácil (menos eficiente): loop dentro del chunk
            for k in range(rr.shape[0]):
                lp = lpips_fn(gen_rgb, rr[k:k+1].repeat(B,1,1,1)).view(-1)  # (B,)
                improve = lp < best_lp
                best_lp = th.where(improve, lp, best_lp)
                best_idx = th.where(improve, th.full_like(best_idx, j+k), best_idx)

        # con idx, calcula SSIM y LPIPS finales con el NN
        nn_gray = real_gray[best_idx]                 # (B,1,H,W)
        nn_rgb  = _as_lpips_rgb(nn_gray)

        lp_final = lpips_fn(gen_rgb, nn_rgb).mean().item()
        ssim_final = ssim_fn((gen_gray + 1)/2, (nn_gray + 1)/2).mean().item()  # SSIM suele ir en [0,1]
        return ssim_final, lp_final

    @th.no_grad()
    def _eval_losses_on_batches(self, data_iter, n_batches: int):
        self.model.eval()
        simple_key = None
        loss_sum = 0.0
        simple_sum = 0.0
        count = 0

        from .resample import UniformSampler
        sampler = UniformSampler(self.diffusion)

        for _ in range(n_batches):
            batch, cond = next(data_iter)
            cond = self.preprocess_input(cond)

            x = batch.to(dist_util.dev())
            model_kwargs = {k: v.to(dist_util.dev()) for k, v in cond.items()}

            t, weights = sampler.sample(x.shape[0], dist_util.dev())
            losses = self.diffusion.training_losses(self.model, x, t, model_kwargs=model_kwargs)

            if simple_key is None:
                simple_key = self._get_loss_simple_key(losses)

            loss = (losses["loss"] * weights).mean().item()
            loss_simple = (losses[simple_key] * weights).mean().item()

            loss_sum += float(loss)
            simple_sum += float(loss_simple)
            count += 1

        self.model.train()
        return loss_sum / max(count, 1), simple_sum / max(count, 1)

    @th.no_grad()
    def run_validation(self):
        if self.val_data is None:
            return None

        # current model
        val_loss, val_loss_simple = self._eval_losses_on_batches(self.val_data, self.val_batches)

        # EMA eval (usa el EMA más “suave”, el último)
        loss_simple_ema = None
        if hasattr(self, "ema_params") and self.ema_params:
            # guarda params actuales
            current_state = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)

            # carga EMA en el modelo
            ema_state = self.mp_trainer.master_params_to_state_dict(self.ema_params[-1])
            self.model.load_state_dict(ema_state, strict=False)

            # eval EMA
            _, loss_simple_ema = self._eval_losses_on_batches(self.val_data, self.val_batches)

            # restaura
            self.model.load_state_dict(current_state, strict=False)

        return val_loss, val_loss_simple, loss_simple_ema
    
    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            #if dist.get_rank() == 0:
            if (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                print("loading model from checkpoint: ", resume_checkpoint)
                self.model.load_state_dict(
                    th.load(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            #if dist.get_rank() == 0:
            if is_rank0():
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = th.load(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

            if self.opt.param_groups[0]['lr'] != self.lr:
                self.opt.param_groups[0]['lr'] = self.lr
    def _swap_to_ema(self):
        if not (hasattr(self, "ema_params") and self.ema_params):
            return None
        current_state = self.mp_trainer.master_params_to_state_dict(self.mp_trainer.master_params)
        ema_state = self.mp_trainer.master_params_to_state_dict(self.ema_params[-1])
        self.model.load_state_dict(ema_state, strict=False)
        return current_state

    def _restore_from_state(self, state_dict):
        if state_dict is not None:
            self.model.load_state_dict(state_dict, strict=False)

    # --- temporal hasta que arreglemos load_data(split='val') ---
    @th.no_grad() 
    def _build_real_pool(self, max_items: int = 512, max_batches: int = 9999):
        if self.val_data is None:
            raise RuntimeError("Necesito val_data para construir el pool fijo de reales.")

        imgs = []
        conds = []
        it = self.val_data  # iterator que ya te llega (ojo: consume)
        # si esto te rompe porque val_data es infinito, ok: paramos por max_items

        while len(imgs) < max_items and max_batches > 0:
            max_batches -= 1
            batch, cond = next(it)
            cond = self.preprocess_input(cond)
            x = batch.to(dist_util.dev()).float()  # [-1,1]

            # guarda 1ch para SSIM y 3ch para LPIPS según quieras
            imgs.append(_as_gray_1ch(x).detach().cpu())
            # para muestrear condicionado, guarda cond (CPU) ya “model_kwargs-ready”
            conds.append({k: v.detach().cpu() for k, v in cond.items()})

        self._real_pool = th.cat(imgs, dim=0)[:max_items]     # (N,1,H,W)
        # cond_pool como lista (más fácil)
        self._cond_pool = []
        for d in conds:
            self._cond_pool.extend([{k: v[i:i+1] for k,v in d.items()} for i in range(list(d.values())[0].shape[0])])
        self._cond_pool = self._cond_pool[:max_items]
    
    @th.no_grad()
    def _sample_batch(self, cond_list, batch_size: int, img_hw: tuple):
        H, W = img_hw
        # arma model_kwargs batch
        model_kwargs = {}
        for k in cond_list[0].keys():
            model_kwargs[k] = th.cat([c[k] for c in cond_list], dim=0).to(dist_util.dev())

        self.model.eval()
        x = self.diffusion.p_sample_loop(
            self.model,
            (batch_size, 3, H, W),          # si luego pasas a gris, ok
            model_kwargs=model_kwargs,
            progress=False
        )
        self.model.train()
        return x  # [-1,1]
    
    ## evaluate metrics
    @th.no_grad()
    def evaluate_metrics(self, global_step: int):
        if self._real_pool is None:
            self._build_real_pool(max_items=cfg.EVAL.REAL_POOL_SIZE)

        # swap EMA
        cur_state = self._swap_to_ema()

        if self._lpips is None:
            self._lpips = LearnedPerceptualImagePatchSimilarity(net_type=cfg.EVAL.LPIPS_NET).to(dist_util.dev())
        if self._ssim is None:
            self._ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(dist_util.dev())
        lpips_fn, ssim_fn = self._lpips, self._ssim

        # sample condicionado usando cond_pool fijo
        B = int(cfg.EVAL.NUM_SAMPLES_METRIC)
        H = W = int(cfg.TRAIN.IMG_SIZE)

        # elige B condiciones fijas (por ejemplo las primeras B)
        cond_list = [ {k: v.to(th.float32) for k,v in self._cond_pool[i].items()} for i in range(B) ]
        gen = self._sample_batch(cond_list, batch_size=B, img_hw=(H,W))  # [-1,1]

        ssim_nn, lpips_nn = self._nn_match_metrics(gen, self._real_pool, lpips_fn, ssim_fn)
        score = float(ssim_nn - float(cfg.EVAL.W_LPIPS) * lpips_nn)

        metrics = {
            "ssim_nn": float(ssim_nn),
            "lpips_nn": float(lpips_nn),
            "score": score,
        }

        # log W&B siempre
        if wandb is not None and getattr(wandb, "run", None) is not None and is_rank0():
            wandb.log({
                "trainer/global_step": global_step,
                "eval/ssim_nn": metrics["ssim_nn"],
                "eval/lpips_nn": metrics["lpips_nn"],
                "eval/score": metrics["score"],
            }, step=global_step)

        self._restore_from_state(cur_state)
        return metrics
    
    ## check_improved + update_best
    def _is_improved(self, score: float):
        best = _json_load(self.best_json_path, default=None)
        if best is None:
            return True
        return score > float(best.get("score", -1e9))

    def update_best(self, global_step: int, metrics: dict):
        # 1) copia checkpoint del step (ya debe existir si estás alineado con save_interval)
        _copy_best_checkpoint(self.output_dir, self.best_dir, global_step, self.ema_rate)

        # 2) escribe best.json
        best_payload = {
            "step": int(global_step),
            "score": float(metrics["score"]),
            "metrics": metrics,
            "paths": {
                "best_dir": self.best_dir,
                "best_eval_dir": self.best_eval_dir,
            },
        }
        _json_save(self.best_json_path, best_payload)

        # 3) log “best/*” a W&B
        if wandb is not None and getattr(wandb, "run", None) is not None and is_rank0():
            wandb.log({
                "best/step": int(global_step),
                "best/score": float(metrics["score"]),
                "best/ssim_nn": float(metrics["ssim_nn"]),
                "best/lpips_nn": float(metrics["lpips_nn"]),
            }, step=global_step)
    ## save_best_eval(overwrite)
    @th.no_grad()
    def save_best_eval(self, global_step: int):
        # borra y recrea best_eval
        if is_rank0():
            _rm_rf(self.best_eval_dir)
            os.makedirs(os.path.join(self.best_eval_dir, "samples"), exist_ok=True)
            os.makedirs(os.path.join(self.best_eval_dir, "conditions"), exist_ok=True)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # swap EMA
        cur_state = self._swap_to_ema()

        H = W = int(cfg.TRAIN.IMG_SIZE)
        N = int(cfg.EVAL.NUM_SAMPLES_SAVE)
        bs = int(cfg.TEST.BATCH_SIZE) if hasattr(cfg, "TEST") else 4
        saved = 0

        while saved < N:
            b = min(bs, N - saved)
            cond_list = []
            for i in range(b):
                idx = (saved + i) % len(self._cond_pool)
                cond_list.append({k: v.to(th.float32) for k, v in self._cond_pool[idx].items()})
            gen = self._sample_batch(cond_list, batch_size=b, img_hw=(H,W))  # [-1,1]
            gen01 = th.clamp((gen + 1) * 0.5, 0, 1).detach().cpu()

            # guarda samples + condiciones (solo rank0)
            if is_rank0():
                for i in range(b):
                    save_image(gen01[i], os.path.join(self.best_eval_dir, "samples", f"sample_{saved+i:05d}.png"))

                    # condición: visualiza máscara argmax de y (si existe)
                    if "y" in cond_list[i]:
                        y = cond_list[i]["y"]  # (1,C,H,W) CPU
                        y_idx = th.argmax(y, dim=1, keepdim=True).float()
                        y_vis = (y_idx / max(float(y_idx.max().item()), 1.0)).repeat(1,3,1,1)
                        save_image(y_vis[0], os.path.join(self.best_eval_dir, "conditions", f"cond_{saved+i:05d}.png"))

            saved += b

        # grid 16
        if is_rank0():
            grid = make_grid(gen01[:min(16, gen01.shape[0])], nrow=8)
            save_image(grid, os.path.join(self.best_eval_dir, "grid.png"))

        self._restore_from_state(cur_state)


    def run_loop(self):
        while (not self.lr_anneal_steps) or (self.step + self.resume_step < self.lr_anneal_steps):
            #global_step = None
            global_step = self.step + self.resume_step

            # --- validation ---
            if self.val_data is not None and self.val_interval and (self.step % self.val_interval == 0):
                out = self.run_validation()
                if out is not None and wandb is not None and getattr(wandb, "run", None) is not None and is_rank0():
                    val_loss, val_loss_simple, val_loss_simple_ema = out
                    log_dict = {
                        "trainer/global_step": global_step,
                        "val/loss": val_loss,
                        "val/loss_simple": val_loss_simple,
                    }
                    if val_loss_simple_ema is not None:
                        log_dict["val/loss_simple_ema"] = val_loss_simple_ema
                    wandb.log(log_dict, step=global_step)
            # --- end validation ---

            logger.log(f"step: {self.step + self.resume_step}")
            #logger.log("ABOUT TO NEXT(self.data)")
            batch, cond = next(self.data)
            #logger.log("GOT BATCH")

            cond = self.preprocess_input(cond)

            if (self.step == 0) and is_rank0():
                # batch está en [-1,1], lo pasamos a [0,1]
                src = ((batch + 1.0) / 2.0).to(dist_util.dev())
                log_inputs_grid_ultrasound(src, cond, self.output_dir, global_step, tag="debug/inputs")

            self.run_step(batch, cond)
            kvs = logger.getkvs()

            if self.step % self.log_interval == 0:
                logger.dumpkvs()

                if wandb is not None and getattr(wandb, "run", None) is not None and is_rank0():
                    wandb.log(
                        {"trainer/global_step": global_step, "train/loss": self.last_train_loss},
                        step=global_step,
                    )

            if self.step > 0 and (self.step % self.save_interval == 0):
                global_step = self.step + self.resume_step

                # A) eval siempre (sin guardar imágenes)
                metrics = None
                if hasattr(cfg, "EVAL") and (global_step % cfg.EVAL.EVERY == 0):
                    metrics = self.evaluate_metrics(global_step)

                # B) ckpt normal + grid monitor
                self.save()
                self.sanity_test(batch=batch, device=dist_util.dev(), cond=cond)

                # C) best: solo si hay métricas (eval step) y mejora
                if metrics is not None and self._is_improved(metrics["score"]):
                    self.update_best(global_step, metrics) # copia ckpt + escribe best.json
                    self.save_best_eval(global_step)  # borra y recrea best_eval/ + guarda 250 PNGs + grid
                    # self.save_best_eval(global_step, metrics)  # esta ya escribe metrics.json dentro
                    # escribir metrics.json fuera como arriba
                    if is_rank0():
                        _json_save(
                            os.path.join(self.best_eval_dir, "metrics.json"),
                            metrics
                        )

            if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                return
            
            self.step += 1
        if (self.step - 1) % self.save_interval != 0:
            self.save()
   
    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)

        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

        # Clear CUDA cache after each step
        torch.cuda.empty_cache()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i: i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]

            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp: 
                losses = compute_losses()

            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            loss = (losses["loss"] * weights).mean()
            self.last_train_loss = float(loss.detach().item())

            if th.isnan(loss).any() or th.isinf(loss).any():
                loss = loss.clamp(-1e6, 1e6)

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self): 
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        logger.logkv("lr", self.opt.param_groups[0]['lr'])
        logger.logkv("lr_anneal_steps", self.lr_anneal_steps)
        logger.logkv("memory_usage", torch.cuda.memory_allocated())

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            #if dist.get_rank() == 0:
            if is_rank0():
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_step):06d}.pt"
                save_path = os.path.join(self.output_dir, filename)
                th.save(state_dict, save_path)
                logger.log(f"saved model {rate} to {save_path}")

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        #if dist.get_rank() == 0:
        if is_rank0():
            optimizer_filename = f"opt{(self.step + self.resume_step):06d}.pt"
            optimizer_path = os.path.join(self.output_dir, optimizer_filename)
            th.save(self.opt.state_dict(), optimizer_path)

        if dist.is_available() and dist.is_initialized():
            dist.barrier()       

    def sanity_test(self, batch, device, cond):
        src_img = ((batch + 1.0) / 2.0).to(device)
        model_kwargs = cond

        # creamos mask_vis desde cond['y'] para visualizar
        y = cond["y"]
        C = y.shape[1]
        mask_cls = th.argmax(y[:, :C, ...], dim=1, keepdim=True).float()  # (B,1,H,W)
        mask_vis = (mask_cls / max(float(C - 1), 1.0)).to(device)  # (B,1,H,W) en [0,1]

        with th.no_grad():
            self.model.eval()
            inference_img, snapshots = self.diffusion.p_sample_loop_with_snapshot(
                self.model,
                (batch.shape[0], 3, batch.shape[2], batch.shape[3]),
                model_kwargs=model_kwargs,
                progress=True
            )
            self.model.train()

        inference_img = (inference_img + 1) / 2.0
        #log_images(inference_img=inference_img, src_img=src_img, snapshots=snapshots, output_dir=self.output_dir, self=self)
        log_images(inference_img=inference_img, src_img=src_img, mask_img=mask_vis, snapshots=snapshots, output_dir=self.output_dir, self=self)


    def preprocess_input(self, data):
        data['label'] = data['label'].long()

        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.num_classes
        input_label = th.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        if 'instance' in data:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = th.cat((input_semantics, instance_edge_map), dim=1)

        if self.drop_rate > 0.0:
            mask = (th.rand([input_semantics.shape[0], 1, 1, 1]) > self.drop_rate).float()
            input_semantics = input_semantics * mask

        cond = {key: value for key, value in data.items() if key not in ['label', 'instance', 'path', 'label_ori']}
        cond['y'] = input_semantics

        return cond

    def get_edges(self, t):
        edge = th.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir(self):
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return self.output_dir #logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{step :06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _to_longpath_windows(path: str) -> str:
    # Permite rutas largas en Windows (si el sistema lo soporta).
    # Para UNC: \\server\share\... -> \\?\UNC\server\share\...
    if os.name != "nt":
        return path
    if path.startswith("\\\\?\\"):
        return path
    if path.startswith("\\\\"):
        return "\\\\?\\UNC\\" + path.lstrip("\\")
    return "\\\\?\\" + path

@th.no_grad()
def log_images(inference_img, src_img, mask_img, snapshots, output_dir, self, max_cols=8):
    """
    Guarda SIEMPRE en local; intenta copiar a NAS si output_dir apunta a NAS.
    """
    B = min(max_cols, src_img.shape[0])

    src = th.clamp(src_img[:B].detach().cpu(), 0, 1)
    fin = th.clamp(inference_img[:B].detach().cpu(), 0, 1)

    m = mask_img[:B].detach().cpu()
    if m.shape[1] == 1:
        m = m.repeat(1, 3, 1, 1)
    m = th.clamp(m, 0, 1)

    ts = sorted(list(snapshots.keys()), reverse=True)
    snaps = [th.clamp(snapshots[t][:B].detach().cpu(), 0, 1) for t in ts]

    rows = [src, m, fin] + snaps
    stack = th.cat(rows, dim=0)
    grid = make_grid(stack, nrow=B)

    step = int(self.step + self.resume_step)
    fname = f"diffusion_results_{step:06d}.png"

    
    # ---- 1) GUARDA LOCAL SIEMPRE ----
    local_dir = os.path.join(os.getcwd(), "qualitative_samples")
    _ensure_dir(local_dir)
    local_path = os.path.join(local_dir, fname)
    save_image(grid, local_path)

    # ---- 2) INTENTA COPIAR A NAS (best effort) ----
    try:
        _ensure_dir(output_dir)
        nas_path = os.path.join(output_dir, fname)
        nas_path_lp = _to_longpath_windows(nas_path)
        shutil.copyfile(local_path, nas_path_lp)
    except Exception as e:
        print(f"[WARN] No pude copiar a NAS: {e}. (Local OK: {local_path})")

    # ---- 3) WANDB: SIEMPRE DESDE LOCAL (evita UNC/latencia/MAX_PATH) ----
    if wandb is not None and getattr(wandb, "run", None) is not None and is_rank0():
        wandb.log(
            {"media/grid": wandb.Image(local_path),
             "trainer/global_step": step},
            step=step,
        )
        
"""
@th.no_grad()
def log_images(inference_img, src_img, mask_img, snapshots, output_dir, self, max_cols=8):
    # src_img: (B,3,H,W) en [0,1]
    # mask_img: (B,1,H,W) en [0,1] (o binaria)
    # inference_img: (B,3,H,W) en [0,1] (raw, NO ema)
    # snapshots: dict {t: (B,3,H,W) en [0,1]}

    B = min(max_cols, src_img.shape[0])

    src = th.clamp(src_img[:B].detach().cpu(), 0, 1)
    fin = th.clamp(inference_img[:B].detach().cpu(), 0, 1)

    m = mask_img[:B].detach().cpu()
    if m.shape[1] == 1:
        m = m.repeat(1, 3, 1, 1)
    m = th.clamp(m, 0, 1)

    # orden fijo de timesteps (descendente)
    ts = sorted(list(snapshots.keys()), reverse=True)
    snaps = [th.clamp(snapshots[t][:B].detach().cpu(), 0, 1) for t in ts]

    # filas: Source, Mask, Final, t=..., t=...
    rows = [src, m, fin] + snaps
    stack = th.cat(rows, dim=0)          # (n_rows*B, 3, H, W)
    grid = make_grid(stack, nrow=B)      # cada fila tiene B imgs

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"diffusion_results_{(self.step+self.resume_step):06d}.png")
    save_image(grid, out_path)

    if wandb is not None and getattr(wandb, "run", None) is not None and is_rank0():
        wandb.log({"media/grid": wandb.Image(out_path),
                   "trainer/global_step": self.step + self.resume_step},
                  step=self.step + self.resume_step)"""

@th.no_grad()
def log_inputs_grid_ultrasound(src_img, cond, output_dir, step, tag="debug/inputs", max_cols=8):
    """
    src_img: (B,3,H,W) en [0,1]
    cond['y']: (B,C,H,W) one-hot (+edges opcional)
    """
    if wandb is None or getattr(wandb, "run", None) is None or not is_rank0():
        return

    # todo a CPU para evitar líos de device
    x = src_img.detach().float().cpu()
    B = min(max_cols, x.shape[0])

    # si por lo que sea viniera en [-1,1], lo arreglamos
    if x.min() < -0.1 or x.max() > 1.1:
        x01 = th.clamp((x[:B] + 1) * 0.5, 0, 1)
    else:
        x01 = th.clamp(x[:B], 0, 1)

    # cond['y'] (B,C,H,W) -> máscara 1ch con argmax -> a 3ch para visualizar
    y = cond["y"][:B].detach()
    y_idx = th.argmax(y, dim=1, keepdim=True).float().cpu()  # (B,1,H,W)
    den = max(float(y_idx.max().item()), 1.0)
    y_vis = (y_idx / den).repeat(1, 3, 1, 1)  # (B,3,H,W)

    grid = make_grid(th.cat([x01, y_vis], dim=0), nrow=B)

    grids_dir = os.path.join(output_dir, "grids")
    os.makedirs(grids_dir, exist_ok=True)
    path = os.path.join(grids_dir, f"inputs_{step:07d}.png")

    save_image(grid, path)
    wandb.log({tag: wandb.Image(path)}, step=step)

def _json_load(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)

def _json_save(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _rm_rf(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)

def _ckpt_files_for_step(output_dir: str, step: int, ema_rates: list):
    files = []
    files.append(os.path.join(output_dir, f"model{step:06d}.pt"))
    for r in ema_rates:
        files.append(os.path.join(output_dir, f"ema_{r}_{step:06d}.pt"))
    files.append(os.path.join(output_dir, f"opt{step:06d}.pt"))
    return files

def _copy_best_checkpoint(output_dir: str, best_dir: str, step: int, ema_rates: list):
    os.makedirs(best_dir, exist_ok=True)
    # limpia best_dir (overwrite)
    for name in os.listdir(best_dir):
        _rm_rf(os.path.join(best_dir, name))
    # copia
    for src in _ckpt_files_for_step(output_dir, step, ema_rates):
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(best_dir, os.path.basename(src)))

def _as_lpips_rgb(x):  # x: (B,C,H,W) in [-1,1]
    if x.shape[1] == 1:
        return x.repeat(1,3,1,1)
    if x.shape[1] == 3:
        return x
    return x[:, :3]  # fallback

def _as_gray_1ch(x):  # x in [-1,1]
    if x.shape[1] == 1:
        return x
    # si realmente es “gris metido en 3ch”, esto lo vuelve 1ch
    return x.mean(dim=1, keepdim=True)
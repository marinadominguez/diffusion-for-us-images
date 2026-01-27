"""
Train a diffusion model on images.
"""

import json
import os
os.environ["MPLBACKEND"] = "Agg"
# opcional pero útil en Windows si algo ya importó matplotlib:
import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib
print("MATPLOTLIB BACKEND:", matplotlib.get_backend())

from argparse import ArgumentParser
from datetime import datetime

from config import cfg, update_config, add_base_args
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    create_model_and_diffusion,
)
from guided_diffusion.train_util import TrainLoop

import wandb
import torch.distributed as dist
import torch as th

import random, numpy as np, torch as th
seed = getattr(cfg, "SEED", 123)
random.seed(seed); np.random.seed(seed); th.manual_seed(seed)
if th.cuda.is_available():
    th.cuda.manual_seed_all(seed)

def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Semantic Diffusion Model')
    add_base_args(parser, cfg) # esto ya mete datadir/savedir y todo
    args = parser.parse_args()
    return args

def main():
    args = get_args_from_command_line()

    update_config(args, cfg)

    exp_name = f"dataset_{cfg.DATASETS.DATASET_MODE}-b_map_min_{cfg.TRAIN.DIFFUSION.B_MAP_MIN}-img_size_{cfg.TRAIN.IMG_SIZE}-lr_{cfg.TRAIN.LR}-diffusion_steps_{cfg.TRAIN.DIFFUSION_STEPS}-b_map_sch_{cfg.TRAIN.B_MAP_SCHEDULER_TYPE}"
    cfg.TRAIN.SAVE_DIR = os.path.join(cfg.TRAIN.SAVE_DIR, exp_name, datetime.now().strftime("%Y-%m-%d"))
    cfg.DATASETS.SAVE_DIR = cfg.TRAIN.SAVE_DIR
    cfg.TRAIN.CHECKPOINT_DIR = cfg.TRAIN.SAVE_DIR

    if not os.path.exists(cfg.TRAIN.SAVE_DIR):
        os.makedirs(cfg.TRAIN.SAVE_DIR, exist_ok=True)

    #dist_util.setup_dist()
    if cfg.TRAIN.DISTRIBUTED_DATA_PARALLEL:
        dist_util.setup_dist()

    def edict_to_dict(x):
        # Converts EasyDict nested to a normal dict (for wandb.config)
        if isinstance(x, dict):
            return {k: edict_to_dict(v) for k, v in x.items()}
        if hasattr(x, '__dict__') and not isinstance(x, (str, int, float, bool, list, tuple)):
            return edict_to_dict(x.__dict__)
        return x
    
    def is_rank0():
        return (not dist.is_available()) or (not dist.is_initialized()) or (dist.get_rank() == 0)

    logger.configure(save_dir=cfg.DATASETS.SAVE_DIR)

    WANDB_LOCAL = r"C:\Users\mdommar1\Dev\DiffusionModels\diffusion-for-us-images"
    os.makedirs(WANDB_LOCAL, exist_ok=True)

    os.environ["WANDB_DIR"] = WANDB_LOCAL
    os.environ["WANDB_CACHE_DIR"] = os.path.join(WANDB_LOCAL, "cache")
    os.environ["WANDB_CONFIG_DIR"] = os.path.join(WANDB_LOCAL, "config")
    os.makedirs(os.environ["WANDB_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["WANDB_CONFIG_DIR"], exist_ok=True)

    # --- W&B init (solo rank 0) ---
    if is_rank0():
        wandb.init(
            project=os.getenv("WANDB_PROJECT", "us-diffusion-benchmark"), #"usdiff_ultrasound"),
            entity=os.getenv("WANDB_ENTITY", None) or None,
            name=os.getenv("WANDB_NAME", exp_name),
            group=os.getenv("WANDB_GROUP", None) or None,
            tags=os.getenv("WANDB_TAGS", "").split(",") if os.getenv("WANDB_TAGS") else None,
            #dir=os.getenv("WANDB_DIR", cfg.DATASETS.SAVE_DIR),
            dir=WANDB_LOCAL,
            config=edict_to_dict(cfg),
            mode=os.getenv("WANDB_MODE", "online"),
            #settings=wandb.Settings(start_method="thread"),  # suele ir mejor en Windows
        )
        wandb.config.update({"save_dir": cfg.DATASETS.SAVE_DIR}, allow_val_change=True)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(cfg)

    if cfg.TRAIN.DISTRIBUTED_DATA_PARALLEL:
        logger.log("Moving model to GPU in a distributed setting...")
        model.to(dist_util.dev())

    else:
        if th.cuda.is_available():
            logger.log("Moving model to CUDA (GPU on a single machine)...")
            model.to("cuda")
        else:
            logger.log("CUDA not available, moving model to CPU...")
            model.to("cpu")

    if cfg.TRAIN.USE_FP16:
        logger.log("Converting model to fp16...")
        model.convert_to_fp16()

    logger.log("creating schedule sampler...")

    schedule_sampler = create_named_schedule_sampler(cfg.TRAIN.SCHEDULE_SAMPLER, diffusion)

    logger.log("creating data loader...")

    train_data = load_data(cfg, split="training")          # infinito
    val_data   = load_data(cfg, split="validation")        # infinito (para val loss)

    # (opcional) para controlar cuanto valida: 
    val_interval=getattr(cfg, "VAL", {}).get("INTERVAL", 200) if hasattr(cfg, "VAL") else 200 # cada X steps
    val_batches=getattr(cfg, "VAL", {}).get("BATCHES", 10) if hasattr(cfg, "VAL") else 10 # sobre cuantos batches promedia

    #val_loader, _ = load_data(cfg, split="validation", return_dataset=True)  # FINITO y determinista

    with open(os.path.join(cfg.DATASETS.SAVE_DIR, 'train_test_config.json'), 'w') as fp:
        json.dump(edict_to_dict(cfg), fp, indent=4)
        fp.close()

    logger.log("training...")

    loop = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=train_data,
        val_data=val_data,   # 👈 IMPORTANTÍSIMO, 
        val_interval=val_interval,
        val_batches=val_batches,
        num_classes=cfg.TRAIN.NUM_CLASSES,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        microbatch=cfg.TRAIN.MICROBATCH,
        lr=cfg.TRAIN.LR,
        ema_rate=cfg.TRAIN.EMA_RATE,
        drop_rate=cfg.TRAIN.DROP_RATE,
        log_interval=cfg.TRAIN.LOG_INTERVAL,
        save_interval=cfg.TRAIN.SAVE_INTERVAL,
        resume_checkpoint=cfg.TRAIN.RESUME_CHECKPOINT,
        use_fp16=cfg.TRAIN.USE_FP16,
        fp16_scale_growth=cfg.TRAIN.FP16_SCALE_GROWTH,
        schedule_sampler=schedule_sampler,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        lr_anneal_steps=cfg.TRAIN.LR_ANNEAL_STEPS,
        output_dir=cfg.TRAIN.CHECKPOINT_DIR,
        distributed_data_parallel=cfg.TRAIN.DISTRIBUTED_DATA_PARALLEL,
    )

    loop.run_loop()


if __name__ == "__main__":
    main()

"""
Train a diffusion model on images.
"""

import json
import os
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


def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Semantic Diffusion Model')
    parser.add_argument('--datadir',
                        default=cfg.DATASETS.DATADIR,
                        type=str)
    parser.add_argument('--savedir',
                        default=cfg.DATASETS.SAVE_DIR,
                        type=str)
    add_base_args(parser, cfg)

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

    dist_util.setup_dist()

    logger.configure(save_dir=cfg.DATASETS.SAVE_DIR)

    logger.log("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(cfg)

    if cfg.TRAIN.DISTRIBUTED_DATA_PARALLEL:
        logger.log("Moving model to GPU in a distributed setting...")
        model.to(dist_util.dev())

    else:
        logger.log("Moving model to CUDA (GPU on a single machine)...")
        model.to('cuda')

    if cfg.TRAIN.USE_FP16:
        logger.log("Converting model to fp16...")
        model.convert_to_fp16()

    logger.log("creating schedule sampler...")

    schedule_sampler = create_named_schedule_sampler(cfg.TRAIN.SCHEDULE_SAMPLER, diffusion)

    logger.log("creating data loader...")

    data = load_data(cfg)

    with open(os.path.join(cfg.DATASETS.SAVE_DIR, 'train_test_config.json'), 'w') as fp:
        json.dump(cfg, fp, indent=4)
        fp.close()

    logger.log("training...")

    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
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
    ).run_loop()


if __name__ == "__main__":
    main()

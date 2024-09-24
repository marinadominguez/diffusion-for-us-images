import argparse

from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.DATASETS = edict()
__C.DATASETS.DATADIR = 'augmented_camus' # '/path/to/dataset' 
__C.DATASETS.SAVE_DIR = 'output/camus/bmaps' # '/path/to/save'
__C.DATASETS.DATASET_MODE = 'camus'

__C.TRAIN = edict()
__C.TRAIN.DIFFUSION = edict()
__C.TRAIN.DIFFUSION.LEARN_SIGMA = True
__C.TRAIN.DIFFUSION.NOISE_SCHEDULE = "cosine"
__C.TRAIN.DIFFUSION.TIMESTEP_RESPACING = ''
__C.TRAIN.DIFFUSION.USE_KL = False
__C.TRAIN.DIFFUSION.PREDICT_XSTART = False 
__C.TRAIN.DIFFUSION.RESCALE_TIMESTEPS = False
__C.TRAIN.DIFFUSION.RESCALE_LEARNED_SIGMAS = False

__C.TRAIN.DIFFUSION.B_MAP_MIN = 0.97 # or 1 if you want to use the original code, without b_maps
__C.TRAIN.DIFFUSION.PRESERVE_LENGTH = False
__C.TRAIN.DIFFUSION.ADD_BUFFER = False
__C.TRAIN.B_MAP_SCHEDULER_TYPE = "cosine"

__C.TRAIN.IMG_SIZE = 128 # 256 if you have enough memory
__C.TRAIN.NUM_CLASSES = 5
__C.TRAIN.LR = 1e-4
__C.TRAIN.ATTENTION_RESOLUTIONS = "32,16,8"
__C.TRAIN.CHANNEL_MULT = None
__C.TRAIN.DROPOUT = 0.0
__C.TRAIN.DIFFUSION_STEPS = 1000 
__C.TRAIN.SCHEDULE_SAMPLER = "uniform"
__C.TRAIN.NUM_CHANNELS = 128 # 256
__C.TRAIN.NUM_HEADS = 1
__C.TRAIN.NUM_HEADS_UPSAMPLE = -1
__C.TRAIN.NUM_HEAD_CHANNELS = 64
__C.TRAIN.NUM_RES_BLOCKS = 2
__C.TRAIN.RESBLOCK_UPDOWN = True
__C.TRAIN.USE_SCALE_SHIFT_NORM = True
__C.TRAIN.USE_CHECKPOINT = True
__C.TRAIN.CLASS_COND = True
__C.TRAIN.WEIGHT_DECAY = 1e-3 
__C.TRAIN.LR_ANNEAL_STEPS = 50000 
__C.TRAIN.BATCH_SIZE = 4 
__C.TRAIN.MICROBATCH = -1
__C.TRAIN.EMA_RATE = "0.9,0.99" 
__C.TRAIN.DROP_RATE = 0.0
__C.TRAIN.LOG_INTERVAL = 100 
__C.TRAIN.SAVE_INTERVAL = 2000 
__C.TRAIN.RESUME_CHECKPOINT = None # optional, if you want to resume training from a checkpoint
__C.TRAIN.USE_FP16 = True
__C.TRAIN.DISTRIBUTED_DATA_PARALLEL = True 
__C.TRAIN.USE_NEW_ATTENTION_ORDER = True
__C.TRAIN.FP16_SCALE_GROWTH = 1e-2 
__C.TRAIN.NUM_WORKERS = 8 
__C.TRAIN.DETERMINISTIC = False
__C.TRAIN.NO_INSTANCE = True
__C.TRAIN.RANDOM_CROP = False
__C.TRAIN.RANDOM_FLIP = False
__C.TRAIN.IS_TRAIN = False

__C.TRAIN.CHECKPOINT_DIR = "output"


__C.TEST = edict()
__C.TEST.S = 1.0
__C.TEST.DETERMINISTIC = True
__C.TEST.INFERENCE_ON_TRAIN = True
__C.TEST.BATCH_SIZE = 4 
__C.TEST.CLIP_DENOISED = True
__C.TEST.NUM_SAMPLES = 1000
__C.TEST.RESULTS_DIR = '/path/to/results'

def update_config(args, _cfg):
    if args.datadir is not None:
        _cfg.DATASETS.DATADIR = args.datadir
    if args.savedir is not None:
        _cfg.TRAIN.SAVE_DIR = args.savedir
    if args.dataset_mode is not None:
        _cfg.DATASETS.DATASET_MODE = args.dataset_mode
    if args.learn_sigma is not None:
        _cfg.TRAIN.DIFFUSION.LEARN_SIGMA = args.learn_sigma
    if args.noise_schedule is not None:
        _cfg.TRAIN.DIFFUSION.NOISE_SCHEDULE = args.noise_schedule
    if args.timestep_respacing is not None:
        _cfg.TRAIN.DIFFUSION.TIMESTEP_RESPACING = args.timestep_respacing
    if args.use_kl is not None:
        _cfg.TRAIN.DIFFUSION.USE_KL = args.use_kl
    if args.predict_xstart is not None:
        _cfg.TRAIN.DIFFUSION.PREDICT_XSTART = args.predict_xstart
    if args.rescale_timesteps is not None:
        _cfg.TRAIN.DIFFUSION.RESCALE_TIMESTEPS = args.rescale_timesteps
    if args.rescale_learned_sigmas is not None:
        _cfg.TRAIN.DIFFUSION.RESCALE_LEARNED_SIGMAS = args.rescale_learned_sigmas
    if args.img_size is not None:
        _cfg.TRAIN.IMG_SIZE = args.img_size
    if args.num_classes is not None:
        _cfg.TRAIN.NUM_CLASSES = args.num_classes
    if args.lr is not None:
        _cfg.TRAIN.LR = args.lr
    if args.attention_resolutions is not None:
        _cfg.TRAIN.ATTENTION_RESOLUTIONS = args.attention_resolutions
    if args.channel_mult is not None:
        _cfg.TRAIN.CHANNEL_MULT = args.channel_mult
    if args.dropout is not None:
        _cfg.TRAIN.DROPOUT = args.dropout
    if args.diffusion_steps is not None:
        _cfg.TRAIN.DIFFUSION_STEPS = args.diffusion_steps
    if args.schedule_sampler is not None:
        _cfg.TRAIN.SCHEDULE_SAMPLER = args.schedule_sampler
    if args.num_channels is not None:
        _cfg.TRAIN.NUM_CHANNELS = args.num_channels
    if args.num_heads is not None:
        _cfg.TRAIN.NUM_HEADS = args.num_heads
    if args.num_heads_upsample is not None:
        _cfg.TRAIN.NUM_HEADS_UPSAMPLE = args.num_heads_upsample
    if args.num_head_channels is not None:
        _cfg.TRAIN.NUM_HEAD_CHANNELS = args.num_head_channels
    if args.num_res_blocks is not None:
        _cfg.TRAIN.NUM_RES_BLOCKS = args.num_res_blocks
    if args.resblock_updown is not None:
        _cfg.TRAIN.RESBLOCK_UPDOWN = args.resblock_updown
    if args.use_scale_shift_norm is not None:
        _cfg.TRAIN.USE_SCALE_SHIFT_NORM = args.use_scale_shift_norm
    if args.use_checkpoint is not None:
        _cfg.TRAIN.USE_CHECKPOINT = args.use_checkpoint
    if args.class_cond is not None:
        _cfg.TRAIN.CLASS_COND = args.class_cond
    if args.weight_decay is not None:
        _cfg.TRAIN.WEIGHT_DECAY = args.weight_decay
    if args.lr_anneal_steps is not None:
        _cfg.TRAIN.LR_ANNEAL_STEPS = args.lr_anneal_steps
    if args.batch_size_train is not None:
        _cfg.TRAIN.BATCH_SIZE = args.batch_size_train
    if args.microbatch is not None:
        _cfg.TRAIN.MICROBATCH = args.microbatch
    if args.ema_rate is not None:
        _cfg.TRAIN.EMA_RATE = args.ema_rate
    if args.drop_rate is not None:
        _cfg.TRAIN.DROP_RATE = args.drop_rate
    if args.log_interval is not None:
        _cfg.TRAIN.LOG_INTERVAL = args.log_interval
    if args.save_interval is not None:
        _cfg.TRAIN.SAVE_INTERVAL = args.save_interval
    if args.resume_checkpoint is not None:
        _cfg.TRAIN.RESUME_CHECKPOINT = args.resume_checkpoint
    if args.use_fp16 is not None:
        _cfg.TRAIN.USE_FP16 = args.use_fp16
    if args.distributed_data_parallel is not None:
        _cfg.TRAIN.DISTRIBUTED_DATA_PARALLEL = args.distributed_data_parallel
    if args.use_new_attention_order is not None:
        _cfg.TRAIN.USE_NEW_ATTENTION_ORDER = args.use_new_attention_order
    if args.fp16_scale_growth is not None:
        _cfg.TRAIN.FP16_SCALE_GROWTH = args.fp16_scale_growth
    if args.num_workers is not None:
        _cfg.TRAIN.NUM_WORKERS = args.num_workers
    if args.no_instance is not None:
        _cfg.TRAIN.NO_INSTANCE = args.no_instance
    if args.deterministic_train is not None:
        _cfg.TRAIN.DETERMINISTIC = args.deterministic_train
    if args.random_crop is not None:
        _cfg.TRAIN.RANDOM_CROP = args.random_crop
    if args.random_flip is not None:
        _cfg.TRAIN.RANDOM_FLIP = args.random_flip
    if args.is_train is not None:
        _cfg.TRAIN.IS_TRAIN = args.is_train
    if args.s is not None:
        _cfg.TEST.S = args.s
    if args.deterministic_test is not None:
        _cfg.TEST.DETERMINISTIC = args.deterministic_test
    if args.inference_on_train is not None:
        _cfg.TEST.INFERENCE_ON_TRAIN = args.inference_on_train
    if args.batch_size_test is not None:
        _cfg.TEST.BATCH_SIZE = args.batch_size_test
    if args.clip_denoised is not None:
        _cfg.TEST.CLIP_DENOISED = args.clip_denoised
    if args.num_samples is not None:
        _cfg.TEST.NUM_SAMPLES = args.num_samples
    if args.results_dir is not None:
        _cfg.TEST.RESULTS_DIR = args.results_dir
    if args.output_dir is not None:
        _cfg.TRAIN.CHECKPOINT_DIR = args.output_dir
    if args.b_map_min is not None:
        _cfg.TRAIN.DIFFUSION.B_MAP_MIN = args.b_map_min
    if args.b_map_scheduler_type is not None:
        _cfg.TRAIN.B_MAP_SCHEDULER_TYPE = args.b_map_scheduler_type
    if args.preserve_length is not None:
        _cfg.TRAIN.DIFFUSION.PRESERVE_LENGTH = args.preserve_length
    if args.add_buffer is not None:
        _cfg.TRAIN.DIFFUSION.ADD_BUFFER = args.add_buffer


def add_base_args(parser, _cfg):
    parser.add_argument('--dataset_mode',
                        default=_cfg.DATASETS.DATASET_MODE,
                        type=str)
    parser.add_argument('--learn_sigma',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=_cfg.TRAIN.DIFFUSION.LEARN_SIGMA)
    parser.add_argument('--noise_schedule',
                        default=_cfg.TRAIN.DIFFUSION.NOISE_SCHEDULE,
                        type=str)
    parser.add_argument('--timestep_respacing',
                        default=_cfg.TRAIN.DIFFUSION.TIMESTEP_RESPACING,
                        type=str)
    parser.add_argument('--use_kl',
                        type=str2bool,
                        nargs='?',
                        const=False,
                        default=_cfg.TRAIN.DIFFUSION.USE_KL)
    parser.add_argument('--predict_xstart',
                        type=str2bool,
                        nargs='?',
                        const=False,
                        default=_cfg.TRAIN.DIFFUSION.PREDICT_XSTART)
    parser.add_argument('--rescale_timesteps',
                        type=str2bool,
                        nargs='?',
                        const=False,
                        default=_cfg.TRAIN.DIFFUSION.RESCALE_TIMESTEPS)
    parser.add_argument('--rescale_learned_sigmas',
                        type=str2bool,
                        nargs='?',
                        const=False,
                        default=_cfg.TRAIN.DIFFUSION.RESCALE_LEARNED_SIGMAS)
    parser.add_argument('--img_size',
                        default=_cfg.TRAIN.IMG_SIZE,
                        type=int)
    parser.add_argument('--num_classes',
                        default=_cfg.TRAIN.NUM_CLASSES,
                        type=int)
    parser.add_argument('--lr',
                        default=_cfg.TRAIN.LR,
                        type=float)
    parser.add_argument('--attention_resolutions',
                        default=_cfg.TRAIN.ATTENTION_RESOLUTIONS,
                        type=str)
    parser.add_argument('--channel_mult',
                        default=_cfg.TRAIN.CHANNEL_MULT,
                        type=int)
    parser.add_argument('--dropout',
                        default=_cfg.TRAIN.DROPOUT,
                        type=float)
    parser.add_argument('--diffusion_steps',
                        default=_cfg.TRAIN.DIFFUSION_STEPS,
                        type=int)
    parser.add_argument('--schedule_sampler',
                        default=_cfg.TRAIN.SCHEDULE_SAMPLER,
                        type=str)
    parser.add_argument('--num_channels',
                        default=_cfg.TRAIN.NUM_CHANNELS,
                        type=int)
    parser.add_argument('--num_heads',
                        default=_cfg.TRAIN.NUM_HEADS,
                        type=int)
    parser.add_argument('--num_heads_upsample',
                        default=_cfg.TRAIN.NUM_HEADS_UPSAMPLE,
                        type=int)
    parser.add_argument('--num_head_channels',
                        default=_cfg.TRAIN.NUM_HEAD_CHANNELS,
                        type=int)
    parser.add_argument('--num_res_blocks',
                        default=_cfg.TRAIN.NUM_RES_BLOCKS,
                        type=int)
    parser.add_argument('--resblock_updown',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=_cfg.TRAIN.RESBLOCK_UPDOWN)
    parser.add_argument('--use_scale_shift_norm',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=_cfg.TRAIN.USE_SCALE_SHIFT_NORM)
    parser.add_argument('--use_checkpoint',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=_cfg.TRAIN.USE_CHECKPOINT)
    parser.add_argument('--class_cond',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=_cfg.TRAIN.CLASS_COND)
    parser.add_argument('--weight_decay',
                        default=_cfg.TRAIN.WEIGHT_DECAY,
                        type=float)
    parser.add_argument('--lr_anneal_steps',
                        default=_cfg.TRAIN.LR_ANNEAL_STEPS,
                        type=int)
    parser.add_argument('--batch_size_train',
                        default=_cfg.TRAIN.BATCH_SIZE,
                        type=int)
    parser.add_argument('--microbatch',
                        default=_cfg.TRAIN.MICROBATCH,
                        type=int)
    parser.add_argument('--ema_rate',
                        default=_cfg.TRAIN.EMA_RATE,
                        type=str)
    parser.add_argument('--drop_rate',
                        default=_cfg.TRAIN.DROP_RATE,
                        type=float)
    parser.add_argument('--log_interval',
                        default=_cfg.TRAIN.LOG_INTERVAL,
                        type=int)
    parser.add_argument('--save_interval',
                        default=_cfg.TRAIN.SAVE_INTERVAL,
                        type=int)
    parser.add_argument('--resume_checkpoint',
                        default=_cfg.TRAIN.RESUME_CHECKPOINT)
    parser.add_argument('--use_fp16', type=str2bool,
                        nargs='?',
                        const=True,
                        default=_cfg.TRAIN.USE_FP16)
    parser.add_argument('--distributed_data_parallel',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=_cfg.TRAIN.DISTRIBUTED_DATA_PARALLEL)
    parser.add_argument('--use_new_attention_order',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=_cfg.TRAIN.USE_NEW_ATTENTION_ORDER)
    parser.add_argument('--fp16_scale_growth',
                        default=cfg.TRAIN.FP16_SCALE_GROWTH,
                        type=float)
    parser.add_argument('--num_workers',
                        default=_cfg.TRAIN.NUM_WORKERS,
                        type=int)
    parser.add_argument('--no_instance',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=_cfg.TRAIN.NO_INSTANCE)
    parser.add_argument('--deterministic_train',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=_cfg.TRAIN.DETERMINISTIC)
    parser.add_argument('--random_crop',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=_cfg.TRAIN.RANDOM_CROP)
    parser.add_argument('--random_flip',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=_cfg.TRAIN.RANDOM_FLIP)
    parser.add_argument('--is_train',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=_cfg.TRAIN.IS_TRAIN)
    parser.add_argument('--s',
                        default=_cfg.TEST.S,
                        type=float)
    parser.add_argument('--deterministic_test',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=_cfg.TEST.DETERMINISTIC)
    parser.add_argument('--inference_on_train',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=_cfg.TEST.INFERENCE_ON_TRAIN)
    parser.add_argument('--batch_size_test',
                        default=_cfg.TEST.BATCH_SIZE,
                        type=int)
    parser.add_argument('--clip_denoised',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=_cfg.TEST.CLIP_DENOISED)
    parser.add_argument('--num_samples',
                        default=_cfg.TEST.NUM_SAMPLES,
                        type=int)
    parser.add_argument('--results_dir',
                        default=_cfg.TEST.RESULTS_DIR,
                        type=str)
    parser.add_argument('--output_dir',
                        default=_cfg.TRAIN.CHECKPOINT_DIR,
                        type=str)

    parser.add_argument('--b_map_min',
                        default=_cfg.TRAIN.DIFFUSION.B_MAP_MIN,
                        type=float)
    parser.add_argument('--b_map_scheduler_type',
                        default=_cfg.TRAIN.B_MAP_SCHEDULER_TYPE,
                        type=str)
    parser.add_argument('--preserve_length',
                        default=_cfg.TRAIN.DIFFUSION.PRESERVE_LENGTH,
                        type=str2bool)
    parser.add_argument('--add_buffer',
                        default=_cfg.TRAIN.DIFFUSION.ADD_BUFFER,
                        type=str2bool)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

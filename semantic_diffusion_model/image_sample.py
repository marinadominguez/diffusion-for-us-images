"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import os
from argparse import ArgumentParser

import numpy as np
import torch as th
import torch.distributed as dist
import torchvision as tv
from PIL import Image
from skimage.color import label2rgb
from skimage.feature import canny

from config import cfg, update_config, add_base_args
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    create_model_and_diffusion,
)

import matplotlib.pyplot as plt
import os

def get_args_from_command_line():
    parser = ArgumentParser(description='Parser of Semantic Diffusion Model')
    parser.add_argument('--datadir',
                        default=cfg.DATASETS.DATADIR)
    parser.add_argument('--savedir',
                        default=cfg.DATASETS.SAVE_DIR)
    add_base_args(parser, cfg)

    args = parser.parse_args()

    return args


def main():
    args = get_args_from_command_line()

    update_config(args, cfg)

    dist_util.setup_dist()
    logger.configure()
    logger.log("creating model and diffusion...")

    th.cuda.empty_cache()

    model, diffusion = create_model_and_diffusion(cfg)

    model_state = th.load(cfg.TRAIN.RESUME_CHECKPOINT, map_location='cpu')
    
    model.load_state_dict(model_state)

    if cfg.TRAIN.DISTRIBUTED_DATA_PARALLEL:
        model.to(dist_util.dev())
        device = dist_util.dev()
    else:
        if th.cuda.is_available():
            th.cuda.set_device(0)
            device = th.cuda.current_device()
        else:
            device = th.device('cpu')
        model.to(device)
        
    logger.log("creating data loader...")
    data = load_data(cfg)

    if cfg.TRAIN.USE_FP16:
        model.convert_to_fp16()

    model.eval()

    if not os.path.exists(cfg.TEST.RESULTS_DIR):
        os.makedirs(cfg.TEST.RESULTS_DIR, exist_ok=True)

    image_path = os.path.join(cfg.TEST.RESULTS_DIR, 'images')
    os.makedirs(image_path, exist_ok=True)
    label_path = os.path.join(cfg.TEST.RESULTS_DIR, 'labels')
    os.makedirs(label_path, exist_ok=True)
    visible_label_path = os.path.join(cfg.TEST.RESULTS_DIR, 'labels_visible')
    os.makedirs(visible_label_path, exist_ok=True)
    inference_path = os.path.join(cfg.TEST.RESULTS_DIR, 'samples')
    os.makedirs(inference_path, exist_ok=True)
    combined_path = os.path.join(cfg.TEST.RESULTS_DIR, 'combined')
    os.makedirs(combined_path, exist_ok=True)


    logger.log("sampling...")
    all_samples = []
    for i, (batch, cond) in enumerate(data):
        src_img = ((batch + 1.0) / 2.0).to(device)
        label_img = (cond['label_ori'].float())
        model_kwargs = preprocess_input(cond, num_classes=cfg.TRAIN.NUM_CLASSES)

        # set hyperparameter
        model_kwargs['s'] = cfg.TEST.S

        sample_fn = (
            diffusion.p_sample_loop_with_snapshot
        )
        inference_img, snapshots = sample_fn(
            model,
            (cfg.TEST.BATCH_SIZE, 3, src_img.shape[2], src_img.shape[3]),
            clip_denoised=cfg.TEST.CLIP_DENOISED,
            model_kwargs=model_kwargs,
            progress=True
        )

        inference_img = (inference_img + 1) / 2.0

        gathered_samples = [th.zeros_like(inference_img) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, inference_img)  # gather not supported with NCCL
        all_samples.extend([sample.cpu().numpy() for sample in gathered_samples])

        for j in range(inference_img.shape[0]):
            tv.utils.save_image(src_img[j],
                                os.path.join(image_path, cond['path'][j].split(os.sep)[-1].split('.')[0] + '.png'))
            tv.utils.save_image(inference_img[j],
                                os.path.join(inference_path, cond['path'][j].split(os.sep)[-1].split('.')[0] + '.png'))
            tv.utils.save_image(label_img[j] / cfg.TRAIN.NUM_CLASSES,
                                os.path.join(visible_label_path,
                                             cond['path'][j].split(os.sep)[-1].split('.')[0] + '.png'))

            label_save_img = Image.fromarray(label_img[j].cpu().detach().numpy()).convert('RGB')
            label_save_img.save(os.path.join(label_path, cond['path'][j].split(os.sep)[-1].split('.')[0] + '.png'))

            src_img_np = src_img[j].permute(1, 2, 0).detach().cpu().numpy()
            label_img_np = label_img[j].repeat(3, 1, 1).permute(1, 2, 0).detach().cpu().numpy()
            inference_img_np = (inference_img[j].permute(1, 2, 0).detach().cpu().numpy())
            inference_img_np = (inference_img_np - np.min(inference_img_np)) / np.ptp(inference_img_np)
            inference_img_np = (255 * (inference_img_np - np.min(inference_img_np)) / np.ptp(inference_img_np)).astype(
                int)
            
            
            combined_imgs = generate_combined_imgs(src_img_np,
                                                   label_img_np.astype(np.int_),
                                                   inference_img_np)

            im = Image.fromarray(combined_imgs)
            im.save(os.path.join(combined_path, cond['path'][j].split(os.sep)[-1].split('.')[0] + '.png'))

        logger.log(f"created {len(all_samples) * cfg.TEST.BATCH_SIZE} samples")     

        log_images(inference_img, label_img, src_img, snapshots=snapshots)

        if len(all_samples) * cfg.TEST.BATCH_SIZE > cfg.TEST.NUM_SAMPLES:
            break

    dist.barrier()
    logger.log("sampling complete")


def log_images(inference_img, label_img, src_img, snapshots=None):

    if snapshots is None:
        snapshots = {}
    num_rows = 3 + len(snapshots)
    num_cols = inference_img.shape[0]
    # Base size for each subplot + some padding
    base_width = 4
    base_height = 4
    fig_width = num_cols * base_width + 2
    fig_height = num_rows * base_height

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    fig.suptitle('Diffusion Model Results', fontsize=16)

    for k in range(num_cols):
        axs[0, k].imshow(src_img[k, 0, ...].cpu().detach().numpy(), cmap='gray')
        axs[0, k].axis('off')

        axs[1, k].imshow(inference_img[k, 0, ...].cpu().detach().numpy(), cmap='gray')
        axs[1, k].axis('off')

        axs[2, k].imshow(label_img[k, ...].cpu().detach().numpy(), cmap='gray')
        axs[2, k].axis('off')

        for i, snap in enumerate(snapshots):
            axs[i + 3, k].imshow(snapshots[snap][k, 0, ...].cpu().detach().numpy(), cmap='gray')
            axs[i + 3, k].axis('off')

    # Set vertical labels for each row outside the loop
    axs[0, 0].set_title("Source Image")
    axs[1, 0].set_title("Inference Image")
    axs[2, 0].set_title("Label Image", pad=20)
    for i, snap in enumerate(snapshots):
        axs[i + 3, 0].set_title(f"Snapshot {snap}")

    plt.tight_layout(rect=[0, 0.0, 1, 0.95])  # Adjust the layout to leave space for the suptitle
    plt.savefig(f"{cfg.TEST.RESULTS_DIR}/sample_{len(os.listdir(cfg.TEST.RESULTS_DIR))}.png")
    plt.close()


def og_generate_combined_imgs(src_in_img, label_in_img, inference_in_img):
    overlayed_label = label2rgb(label=label_in_img[:, :, 0], image=inference_in_img,
                                bg_label=0,
                                channel_axis=-1,
                                alpha=0.2, image_alpha=1)

    src_out_img = (src_in_img * 255).astype('uint8')
    overlayed_label = (overlayed_label * 255).astype('uint8')

    edges = canny(label_in_img[:, :, 0] / label_in_img[:, :, 0].max())
    edges = np.expand_dims(edges, axis=-1)
    edges = np.concatenate((edges, edges, edges), axis=-1) * 255
    edges[:, :, 2] = 0

    overlayed_edge_label = np.copy(inference_in_img)
    overlayed_edge_label[edges == 255] = 255

    combined_imgs = np.concatenate((src_out_img, inference_in_img, overlayed_label, overlayed_edge_label),
                                   axis=0).astype(
        np.uint8)

    return combined_imgs

def generate_combined_imgs(src_in_img, label_in_img, inference_in_img):
    # label in img is already 3 channels
    label_rgb_img = label_in_img
    
    # Convert source to uint8
    src_out_img = (src_in_img * 255).astype('uint8')
    
    # Normalize label_rgb_img to [0, 255]
    label_rgb_img = (label_rgb_img / label_rgb_img.max() * 255).astype('uint8')
    
    # Blend the label image with the inference image
    alpha = 0.2
    blended_img = (alpha * label_rgb_img + (1 - alpha) * inference_in_img).astype('uint8')
    
    # Perform edge detection on the label image
    edges = canny(label_in_img[:, :, 0] / label_in_img[:, :, 0].max())
    edges = np.expand_dims(edges, axis=-1)
    edges = np.concatenate((edges, edges, edges), axis=-1) * 255
    edges[:, :, 2] = 0
    
    # Overlay edges on the inference image
    overlayed_edge_label = np.copy(inference_in_img)
    overlayed_edge_label[edges == 255] = 255
    
    # Combine images vertically
    combined_imgs = np.concatenate((src_out_img, inference_in_img, label_rgb_img, overlayed_edge_label), axis=0).astype(np.uint8)
    
    return combined_imgs


def preprocess_input(data, num_classes):
    # move to GPU and change data types
    data['label'] = data['label'].long()

    # create one-hot label map
    label_map = data['label']
    bs, _, h, w = label_map.size()
    input_label = th.FloatTensor(bs, num_classes, h, w).zero_()
    input_semantics = input_label.scatter_(1, label_map, 1.0)

    # concatenate instance map if it exists
    if 'instance' in data:
        inst_map = data['instance']
        instance_edge_map = get_edges(inst_map)
        input_semantics = th.cat((input_semantics, instance_edge_map), dim=1)

    return {'y': input_semantics}

def get_edges(t):
    edge = th.ByteTensor(t.size()).zero_()
    edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
    edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
    return edge.float()

if __name__ == "__main__":
    main()
    
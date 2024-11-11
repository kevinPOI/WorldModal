#!/usr/bin/env python3

"""
Script to decode tokenized video into images/video.
Example usage: See https://github.com/1x-technologies/1xgpt?tab=readme-ov-file#1x-genie-baseline
"""
import torchvision.transforms as transforms
import argparse
import math
import os
from PIL import Image, ImageDraw
import cv2
import numpy as np
import torch
import torch.distributed.optim
import torch.utils.checkpoint
import torch.utils.data
import torchvision.transforms.v2.functional as transforms_f
from einops import rearrange
from matplotlib import pyplot as plt

from data import RawTokenDataset
from magvit2.config import VQConfig
from magvit2.models.lfqgan import VQModel


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize tokenized video as GIF or comic.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame skip",
    )
    parser.add_argument(
        "--token_dir",
        type=str,
        default="data/genie_generated",
        help="Directory of tokens, in the format of `video.bin` and `metadata.json`. "
             "Visualized gif and comic will be written here.",
    )
    parser.add_argument(
        "--offset", type=int, default=0, help="Offset to start generating images from"
    )
    parser.add_argument(
        "--fps", type=int, default=2, help="Frames per second"
    )
    parser.add_argument(
        "--max_images", type=int, default=None, help="Maximum number of images to generate. None for all."
    )
    parser.add_argument(
        "--disable_comic", action="store_true",
        help="Comic generation assumes `token_dir` follows the same format as generate: e.g., "
             "`prompt | predictions | gtruth` in `video.bin`, `window_size` in `metadata.json`."
             "Therefore, comic should be disabled when visualizing videos without this format, such as the dataset."
    )
    args = parser.parse_args()

    return args


# def export_to_gif(frames: list, output_gif_path: str, fps: int):
#     """
#     Export a list of frames to a GIF.

#     Args:
#     - frames (list): List of frames (as numpy arrays or PIL Image objects).
#     - output_gif_path (str): Path to save the output GIF.
#     - fps (int): Desired frames per second.
#     """
#     # Convert numpy arrays to PIL Images if needed
#     pil_frames = [Image.fromarray(frame) if isinstance(
#         frame, np.ndarray) else frame for frame in frames]

#     duration_ms = 1000 / fps
#     pil_frames[0].save(output_gif_path.replace(".mp4", ".gif"),
#                        format="GIF",
#                        append_images=pil_frames[1:],
#                        save_all=True,
#                        duration=duration_ms,
#                        loop=0)
def decode_latents_wrapper(batch_size=16, tokenizer_ckpt="data/magvit2.ckpt", max_images=None):
    device = "cuda"
    dtype = torch.bfloat16

    model_config = VQConfig()
    model = VQModel(model_config, ckpt_path=tokenizer_ckpt)
    model = model.to(device=device, dtype=dtype)

    @torch.no_grad()
    def decode_latents(video_data):
        """
        video_data: (b, h, w), where b is different from training/eval batch size.
        """
        decoded_imgs = []

        for shard_ind in range(math.ceil(len(video_data) / batch_size)):
            batch = torch.from_numpy(video_data[shard_ind * batch_size: (shard_ind + 1) * batch_size].astype(np.int64))
            if model.use_ema:
                with model.ema_scope():
                    quant = model.quantize.get_codebook_entry(rearrange(batch, "b h w -> b (h w)"),
                                                              bhwc=batch.shape + (model.quantize.codebook_dim,)).flip(1)
                    decoded_imgs.append(((rescale_magvit_output(model.decode(quant.to(device=device, dtype=dtype))))))
            if max_images and len(decoded_imgs) * batch_size >= max_images:
                break

        return [transforms_f.to_pil_image(img) for img in torch.cat(decoded_imgs)]

    return decode_latents
def encode_img_test(tokenizer_ckpt="data/magvit2.ckpt"):
    device = "cuda"
    dtype = torch.bfloat16

    model_config = VQConfig()
    model = VQModel(model_config, ckpt_path=tokenizer_ckpt)
    model = model.to(device=device, dtype=dtype)
    image = Image.open('c1.jpg').convert('RGB')  # Ensures 3 channels
    cv_image = np.array(image)
    batch_size = 1
    bhwc = torch.Size([batch_size, 16,16, model.quantize.codebook_dim])
# Convert RGB to BGR format for OpenCV
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    restore_transform = transforms.Resize((image.size[1], image.size[0]))
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device = device, dtype = dtype)
    quant, emb_loss, info, loss_breakdown = model.encode(image_tensor)
    uint_enc = model.quantize.quant_to_int(quant.flip(1), bhwc)
    if True:
        new_quant = model.quantize.get_codebook_entry(rearrange(uint_enc, "b h w -> b (h w)"),
                                                              bhwc=bhwc).flip(1)
        pass
    decoded =restore_transform(rescale_magvit_output2(model.decode(quant)))
#     decoded_image = decoded.detach().cpu().to(torch.float32)[0]
#     decoded_image = restore_transform(decoded_image)
# # Convert the tensor from CxHxW to HxWxC for displaying
#     if decoded_image.shape[0] == 3:  # Assuming a 3-channel image
#         decoded_image = decoded_image.permute(1, 2, 0)

#     # Convert from [0, 1] range if necessary (for float images)
#     decoded_image = decoded_image * 255

#     # Convert to numpy
#     decoded_image = decoded_image.numpy().astype('uint8')
    decoded_image = decoded[0]
    if decoded_image.shape[0] == 3:  # Assuming a 3-channel image
        decoded_image = decoded_image.permute(1, 2, 0)
    
    # Display
    decoded_image = cv2.cvtColor(decoded_image.numpy(), cv2.COLOR_RGB2BGR)
    cv2.imshow('Decoded Image', decoded_image)
    cv2.imshow('Original Image', cv_image)
    cv2.waitKey(0)
def encode_img(tokenizer_ckpt="data/magvit2.ckpt"):
    device = "cuda"
    dtype = torch.bfloat16

    model_config = VQConfig()
    model = VQModel(model_config, ckpt_path=tokenizer_ckpt)
    model = model.to(device=device, dtype=dtype)
    image = Image.open('c1.jpg').convert('RGB')  # Ensures 3 channels
    cv_image = np.array(image)

# Convert RGB to BGR format for OpenCV
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    restore_transform = transforms.Resize((image.size[1], image.size[0]))
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device = device, dtype = dtype)
    quant, emb_loss, info, loss_breakdown = model.encode(image_tensor)
    decoded =restore_transform(rescale_magvit_output2(model.decode(quant)))
#     decoded_image = decoded.detach().cpu().to(torch.float32)[0]
#     decoded_image = restore_transform(decoded_image)
# # Convert the tensor from CxHxW to HxWxC for displaying
#     if decoded_image.shape[0] == 3:  # Assuming a 3-channel image
#         decoded_image = decoded_image.permute(1, 2, 0)

#     # Convert from [0, 1] range if necessary (for float images)
#     decoded_image = decoded_image * 255

#     # Convert to numpy
#     decoded_image = decoded_image.numpy().astype('uint8')
    decoded_image = decoded[0]
    if decoded_image.shape[0] == 3:  # Assuming a 3-channel image
        decoded_image = decoded_image.permute(1, 2, 0)
    
    # Display
    decoded_image = cv2.cvtColor(decoded_image.numpy(), cv2.COLOR_RGB2BGR)
    cv2.imshow('Decoded Image', decoded_image)
    cv2.imshow('Original Image', cv_image)
    cv2.waitKey(0)
    # plt.imshow(image)
    # plt.imshow(decoded_image)
    # plt.axis('off')  # Turn off axis
    # plt.show()
def rescale_magvit_output2(magvit_output):
    """
    [min, max] -> [0, 255]

    Important: clip to [0, 255]
    """
    min_range = -0.3
    max_range = 1
    r = max_range - min_range
    rescaled_output = 255 * (magvit_output.detach().cpu() -  min_range) / r
    clipped_output = torch.clamp(rescaled_output, 0, 255).to(dtype=torch.uint8)
    return clipped_output



@torch.no_grad()
def main():
    #args = parse_args()
    encode_img2()
    # Load tokens
    

if __name__ == "__main__":
    main()

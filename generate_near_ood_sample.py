import os
from argparse import ArgumentParser

import torch
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusion_inpainting_pipeline import StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor

from datasets import CleanedImageNet

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, choices=('cpu', 'cuda'))
    parser.add_argument("--lr", type=float, default=2e11)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num_images_per_prompt", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--iter", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--strength", type=float, default=0.7)
    parser.add_argument("--guidance_scale", type=float, default=8)
    parser.add_argument("--sigmas", type=int, default=None)
    parser.add_argument("--crops_coords", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--resize_mode", type=str, default='default')
    parser.add_argument("--data-dir", type=str, default='path/to/imagenet')
    parser.add_argument("--cleaned-dir", type=str, default='cleaned/')
    parser.add_argument("--save-dir", type=str, default='path/to/save')
    parser.add_argument("--context-dir", type=str, default='texts/matched_info.jsonl')
    parser.add_argument("--method", type=str, default='softmax')

    args = parser.parse_args()
    return args


@torch.no_grad()
def prepare_latents(pipe, init_image, generator, strength, sigmas, bs, num_inference_steps, timesteps,
                    height, width, device, crops_coords, resize_mode):
    images = pipe.image_processor.preprocess(init_image, height=height, 
                                            width=width, crops_coords=crops_coords, 
                                            resize_mode=resize_mode).to(device)
    _, noise, image_latents = pipe.prepare_latents(bs, 4, height, width, images.dtype, images.device, generator, 
                                                    image=images, 
                                                    return_noise=True, 
                                                    return_image_latents=True)

    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, device, timesteps, sigmas
    )

    timesteps, num_inference_steps = pipe.get_timesteps(
        num_inference_steps,
        strength,
        device,
        # denoising_start=None,
    )
    return images, image_latents, noise, timesteps


def get_loss(model, images, methods, device):
    logits = model(images)

    if methods == 'msp':
        logits = torch.nn.functional.softmax(model(images), dim=1)
        loss    = torch.nn.functional.kl_div(logits, torch.ones(1000).to(device) / 1000.0)
    elif methods == 'energy':
        logits = logits.mean(1) - torch.logsumexp(logits, dim=1)
        # loss    = torch.pow(torch.nn.functional.relu(-4 - logits), 2).mean()
        loss    = torch.pow(-5 - logits, 2).mean()
        print(f'logits: {logits.tolist()}, Loss: {loss.item()}')

    return loss


def train():
    args = parse_args()

    dataset    = CleanedImageNet(args.data_dir, args.cleaned_dir, args.context_dir, args.height, args.width, transform=None)
    dataloader = torch.utils.data.DataLoader(dataset, args.batch_size, shuffle=False, num_workers=8, 
                                             pin_memory=True, collate_fn=dataset.collate_fn)
    print(f'Length of dataset: {len(dataset)}')
    pipe       = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting").to(args.device)
    generator  = torch.Generator(device="cuda").manual_seed(0)

    backbone  = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(args.device)
    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    backbone.eval()
    
    for idx, (init_images, mask_images, prompts, cates, files) in tqdm(enumerate(dataloader), total=len(dataloader)):
        bs = len(init_images)

        with torch.no_grad():
            images, image_latents, noises, timesteps = prepare_latents(
                pipe, init_images, generator, args.strength,
                args.sigmas, bs, args.num_inference_steps, args.timesteps, 
                args.height, args.width, pipe.unet.device, args.crops_coords, args.resize_mode)

        for i in range(args.iter):
            noise_latents = pipe.scheduler.add_noise(image_latents, noises, timesteps[:1])

            with torch.no_grad():
                mask = pipe.mask_processor.preprocess(
                    mask_images, height=args.height, width=args.width, 
                    resize_mode=args.resize_mode, crops_coords=args.crops_coords
                    ).to(args.device)
                masked_image_latents = images * (mask < 0.5)

                latents = pipe(
                prompt=prompts,
                # image=init_image,
                mask_image=mask_images,
                latents=noise_latents,
                masked_image_latents=masked_image_latents,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,  # steps between 15 and 30 work well for us
                strength=args.strength,  # make sure to use `strength` below 1.0
                generator=generator,
                output_type="latent"
                ).images

                has_latents_mean = hasattr(pipe.vae.config, "latents_mean") and pipe.vae.config.latents_mean is not None
                has_latents_std = hasattr(pipe.vae.config, "latents_std") and pipe.vae.config.latents_std is not None

                if has_latents_mean and has_latents_std:
                    latents_mean = (torch.tensor(pipe.vae.config.latents_mean
                                                ).view(1, 4, 1, 1).to(latents.device, latents.dtype))
                    latents_std  = (torch.tensor(pipe.vae.config.latents_std
                                                ).view(1, 4, 1, 1).to(latents.device, latents.dtype))
                    latents = latents * latents_std / pipe.vae.config.scaling_factor + latents_mean
                else:
                    latents = latents / pipe.vae.config.scaling_factor

                if args.iter == 1:
                    image_gen = pipe.vae.decode(latents, return_dict=False)[0]
                    image_gen = (image_gen / 2 + 0.5).clamp(0, 1)
                    break

            with torch.enable_grad():
                latents.requires_grad_(True)

                image_gen = pipe.vae.decode(latents, return_dict=False)[0]
                image_gen = (image_gen / 2 + 0.5).clamp(0, 1)

                image_in = transform(image_gen).to(args.device)
                loss     = get_loss(backbone, image_in, args.method, args.device)
                loss.backward()
                noises = noises - latents.grad * args.lr
            # if latents.grad is not None:
            #     latents.grad.zero_()

        with torch.no_grad():
            image_gen = (image_gen * 255).round().detach()
            image_gen = image_gen.cpu().to(torch.uint8).permute(0, 2, 3, 1).numpy()

            for i in range(bs):
                save_dir = os.path.join(args.save_dir, cates[i])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                img = Image.fromarray(image_gen[i])
                name, format = files[i].split('.')
                img.save(os.path.join(args.save_dir, cates[i], f'{name}_ood.{format}'))


if __name__ == "__main__":
    train()
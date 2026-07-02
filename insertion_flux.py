#!/usr/bin/env python3
import argparse
from pathlib import Path

import rembg
import torch
from diffusers import FluxFillPipeline
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-Fill-dev")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mask-invert", action="store_true")
    args = parser.parse_args()

    image_path = args.cache_dir / "origin_rendering.png"
    mask_path = args.cache_dir / "mask.png"
    out_path = args.cache_dir / "inpainted.png"
    removed_bg_path = args.cache_dir / "removed_bg.png"

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.NEAREST)

    if args.mask_invert:
        mask = Image.eval(mask, lambda v: 255 - v)

    pipe = FluxFillPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    result = pipe(
        prompt=args.prompt,
        image=image,
        mask_image=mask,
        height=image.height,
        width=image.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]

    result.save(out_path)

    removed_bg = rembg.remove(result)
    removed_bg.save(removed_bg_path)

    print("saved:", out_path)
    print("saved:", removed_bg_path)


if __name__ == "__main__":
    main()
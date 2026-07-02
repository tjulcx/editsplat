#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from diffusers import FluxFillPipeline
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-base", type=Path, required=True)
    p.add_argument("--seg-prompt", type=str, required=True)
    p.add_argument("--inpaint-scale", type=str, default="1")
    p.add_argument("--view-num", type=str, default="96")
    p.add_argument("--prompt", type=str, default="clean realistic background, object removed")
    p.add_argument("--model", type=str, default="black-forest-labs/FLUX.1-Fill-dev")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance-scale", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--mask-invert", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    image_dir = args.cache_base / f"{args.seg_prompt}_pruned_{args.view_num}_view"
    mask_dir = args.cache_base / f"{args.seg_prompt}_pruned_mask_scale_{args.inpaint_scale}_{args.view_num}_view"
    out_dir = args.cache_base / f"{args.seg_prompt}_scale_{args.inpaint_scale}_{args.view_num}_view_inpaint_ctn"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(image_dir.glob("*.png"))
    if args.limit:
        images = images[: args.limit]

    pipe = FluxFillPipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload()

    for i, image_path in enumerate(images):
        out_path = out_dir / image_path.name
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path.name}")
            continue

        mask_path = mask_dir / image_path.name
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.NEAREST)

        if args.mask_invert:
            mask = Image.eval(mask, lambda v: 255 - v)

        generator = torch.Generator(device="cuda").manual_seed(args.seed)

        print(f"[{i+1}/{len(images)}] {image_path.name}")
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

    print("done:", out_dir)


if __name__ == "__main__":
    main()    main()
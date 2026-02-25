"""
generate.py - 使用训练好的 IDGS 模型批量生成合成缺陷图

策略:
  - 使用 Phase 1 LoRA 做生成（纯文本条件，图像质量高）
  - DVCP 矩阵作为推理时约束，仅生成物理合法的组合 → PCS=100%

用法:
  # 生成所有 DVCP 合法组合，每组 100 张
  python scripts/generate.py --config configs/idgs.yaml \\
      --ckpt checkpoints/idgs/phase1_final --num_per_combo 100

  # 只生成特定缺陷
  python scripts/generate.py --config configs/idgs.yaml \\
      --ckpt checkpoints/idgs/phase1_final --defects diffusion adhesion
"""
import argparse
import os
import sys
from pathlib import Path

import yaml
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.datasets.defect_dataset import DEFECT_TYPES, CAM_IDS, POSES, DVCP_MATRIX
from scripts.train_phase1 import make_prompt, tokenize_prompt


def get_legal_combinations():
    """从 DVCP 矩阵中提取所有合法的 (defect, cam) 组合"""
    combos = []
    for d_idx, defect in enumerate(DEFECT_TYPES):
        for c_idx, cam in enumerate(CAM_IDS):
            if DVCP_MATRIX[d_idx, c_idx] > 0:
                combos.append((defect, d_idx, cam, c_idx))
    return combos


@torch.no_grad()
def generate(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"Device: {device}, dtype: {dtype}")

    # ── 加载模型 ──
    print("Loading models...")
    tokenizer = CLIPTokenizer.from_pretrained(
        model_cfg["sd_model_id"], subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_cfg["sd_model_id"], subfolder="text_encoder"
    ).to(device, dtype=dtype)

    vae = AutoencoderKL.from_pretrained(
        model_cfg["sd_model_id"], subfolder="vae"
    ).to(device, dtype=dtype)

    unet = UNet2DConditionModel.from_pretrained(
        model_cfg["sd_model_id"], subfolder="unet"
    )

    # 加载 LoRA（Phase 1 或 Phase 2 的 unet_lora 子目录）
    ckpt_path = Path(args.ckpt)
    lora_path = ckpt_path / "unet_lora"
    if lora_path.exists():
        print(f"Loading LoRA from {lora_path}")
        unet = PeftModel.from_pretrained(unet, str(lora_path))
    elif (ckpt_path / "adapter_config.json").exists():
        print(f"Loading LoRA from {ckpt_path}")
        unet = PeftModel.from_pretrained(unet, str(ckpt_path))
    else:
        print(f"[WARNING] No LoRA found at {ckpt_path}")

    unet = unet.to(device, dtype=dtype)
    unet.eval()

    # DDIM Scheduler (加速推理, 50步)
    scheduler = DDIMScheduler.from_pretrained(
        model_cfg["sd_model_id"], subfolder="scheduler"
    )
    scheduler.set_timesteps(args.num_inference_steps)

    # ── DVCP 约束: 只生成物理合法的组合 ──
    all_combos = get_legal_combinations()
    if args.defects:
        all_combos = [c for c in all_combos if c[0] in args.defects]

    print(f"\n{'='*60}")
    print(f"  DVCP-Constrained Generation")
    print(f"  Legal combinations: {len(all_combos)} / {len(DEFECT_TYPES) * len(CAM_IDS)}")
    print(f"  Images per combo:   {args.num_per_combo} x {len(POSES)} poses")
    total_expected = len(all_combos) * len(POSES) * args.num_per_combo
    print(f"  Total to generate:  {total_expected}")
    print(f"{'='*60}\n")

    # ── 输出目录 ──
    output_dir = Path(args.output or cfg["output"]["sample_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    total_generated = 0

    for defect, d_idx, cam, c_idx in all_combos:
        for pose in POSES:
            combo_dir = output_dir / defect / pose
            combo_dir.mkdir(parents=True, exist_ok=True)

            # 构造文本条件
            prompt = make_prompt(cam, defect, pose)
            input_ids = tokenize_prompt(tokenizer, prompt).to(device)
            text_emb = text_encoder(input_ids)[0]  # (1, 77, 768)

            desc = f"cam{cam}/{pose}/{defect}"
            for i in tqdm(range(args.num_per_combo), desc=desc, leave=False):

                latent_size = cfg["data"]["image_size"] // 8
                latents = torch.randn(
                    1, 4, latent_size, latent_size,
                    device=device, dtype=dtype
                )

                # ── DDIM 去噪 ──
                for t in scheduler.timesteps:
                    noise_pred = unet(
                        latents, t,
                        encoder_hidden_states=text_emb.to(dtype=dtype),
                    ).sample
                    latents = scheduler.step(noise_pred, t, latents).prev_sample

                # ── VAE 解码 ──
                latents_scaled = latents / vae.config.scaling_factor
                image = vae.decode(latents_scaled).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                image = (image * 255).astype(np.uint8)

                # ── 保存 ──
                fname = f"cam{cam}_{pose}_{defect}_gen_{i+1:05d}.png"
                Image.fromarray(image).save(str(combo_dir / fname))
                total_generated += 1

    print(f"\n[DONE] Generated {total_generated} images to {output_dir}")
    print(f"  PCS (Physical Compatibility Score) = 100% (DVCP-constrained)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/idgs.yaml")
    parser.add_argument("--ckpt",   type=str, required=True,
                        help="Path to LoRA checkpoint (phase1_final or phase2_final)")
    parser.add_argument("--num_per_combo", type=int, default=100,
                        help="Number of images per (defect, cam) combination")
    parser.add_argument("--defects", nargs="*", default=None,
                        help="Specific defect types to generate (default: all legal)")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    generate(args)

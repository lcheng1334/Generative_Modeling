"""
generate.py - 使用训练好的 IDGS 模型批量生成合成缺陷图

用法:
  # 生成所有合法 (cam, defect) 组合，每组 100 张
  python scripts/generate.py --config configs/idgs.yaml \\
      --ckpt checkpoints/idgs/phase2_final --num_per_combo 100

  # 只生成特定缺陷
  python scripts/generate.py --config configs/idgs.yaml \\
      --ckpt checkpoints/idgs/phase2_final --defects diffusion adhesion

  # 指定严重程度
  python scripts/generate.py --config configs/idgs.yaml \\
      --ckpt checkpoints/idgs/phase2_final --severity 0.8
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
    DDPMScheduler,
    UNet2DConditionModel,
    DDIMScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.datasets.defect_dataset import DEFECT_TYPES, CAM_IDS, POSES, DVCP_MATRIX
from src.models.dvcp_module import DVCPConditioner
from src.models.san_module import SANWrapper
from src.models.severity_estimator import SeverityEncoder
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

    # 加载 LoRA
    ckpt_path = Path(args.ckpt)
    lora_path = ckpt_path / "unet_lora"
    if lora_path.exists():
        print(f"Loading LoRA from {lora_path}")
        unet = PeftModel.from_pretrained(unet, str(lora_path))
    unet = unet.to(device, dtype=dtype)
    unet.eval()

    # 加载 DVCP Conditioner
    dvcp_cond = DVCPConditioner()
    dvcp_cond_path = ckpt_path / "dvcp_conditioner.pt"
    if dvcp_cond_path.exists():
        dvcp_cond.load_state_dict(torch.load(str(dvcp_cond_path), map_location="cpu"))
        print("Loaded DVCP Conditioner")
    dvcp_cond = dvcp_cond.to(device)
    dvcp_cond.eval()

    # 加载 SAN
    san = SANWrapper(feature_dims=[4, 320, 640, 1280])
    san_path = ckpt_path / "san_wrapper.pt"
    if san_path.exists():
        san.load_state_dict(torch.load(str(san_path), map_location="cpu"))
        print("Loaded SAN")
    san = san.to(device)
    san.eval()

    # 加载 Severity Encoder
    sev_enc = SeverityEncoder()
    # severity encoder 如果训练了就加载, 否则用初始化的
    sev_enc = sev_enc.to(device)
    sev_enc.eval()

    # DDIM Scheduler (加速推理, 50步)
    scheduler = DDIMScheduler.from_pretrained(
        model_cfg["sd_model_id"], subfolder="scheduler"
    )
    scheduler.set_timesteps(args.num_inference_steps)

    # ── 输出目录 ──
    output_dir = Path(args.output or cfg["output"]["sample_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 确定要生成的组合 ──
    all_combos = get_legal_combinations()
    if args.defects:
        all_combos = [c for c in all_combos if c[0] in args.defects]

    print(f"\nWill generate for {len(all_combos)} (defect, cam) combinations")
    print(f"  {args.num_per_combo} images per combo x {len(POSES)} poses")

    total_generated = 0

    for defect, d_idx, cam, c_idx in all_combos:
        for pose in POSES:
            pose_idx = 0 if pose == "p0" else 1

            combo_dir = output_dir / defect / pose
            combo_dir.mkdir(parents=True, exist_ok=True)

            prompt = make_prompt(cam, defect, pose)
            input_ids = tokenize_prompt(tokenizer, prompt).to(device)

            desc = f"cam{cam}/{pose}/{defect}"
            for i in tqdm(range(args.num_per_combo), desc=desc, leave=False):

                # ── 文本条件 ──
                text_emb = text_encoder(input_ids)[0]  # (1, seq_len, 768)

                # ── DVCP 条件 ──
                dvcp_emb = dvcp_cond(
                    cam_idx=torch.tensor([c_idx], device=device),
                    defect_idx=torch.tensor([d_idx], device=device),
                    pose_idx=torch.tensor([pose_idx], device=device),
                )  # (1, 1, 768)

                # ── Severity 条件 ──
                if args.severity is not None:
                    sev_val = torch.tensor([args.severity], device=device)
                else:
                    # 随机采样严重程度 (mild~severe)
                    sev_val = torch.rand(1, device=device) * 0.8 + 0.1
                sev_emb = sev_enc(sev_val).to(dtype=dtype)  # (1, 1, 768)

                # ── 合并条件 (和 train_phase2 保持一致: text + dvcp) ──
                encoder_hidden_states = torch.cat([
                    text_emb.to(dtype=dtype),
                    dvcp_emb.to(dtype=dtype),
                ], dim=1)

                # ── 去噪 ──
                latent_size = cfg["data"]["image_size"] // 8
                latents = torch.randn(
                    1, 4, latent_size, latent_size,
                    device=device, dtype=dtype
                )

                # NOTE: SAN 是训练时对 VAE latent 的归一化，推理时初始噪声不需要
                # SAN 的效果已经融入了 LoRA 权重中

                for t in scheduler.timesteps:
                    noise_pred = unet(
                        latents, t,
                        encoder_hidden_states=encoder_hidden_states,
                    ).sample
                    latents = scheduler.step(noise_pred, t, latents).prev_sample

                # ── 解码 ──
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/idgs.yaml")
    parser.add_argument("--ckpt",   type=str, required=True,
                        help="Path to Phase 2 checkpoint")
    parser.add_argument("--num_per_combo", type=int, default=100,
                        help="Number of images per (defect, cam) combination")
    parser.add_argument("--defects", nargs="*", default=None,
                        help="Specific defect types to generate (default: all legal)")
    parser.add_argument("--severity", type=float, default=None,
                        help="Fixed severity level (0-1). Default: random sampling")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    generate(args)

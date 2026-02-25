"""
train_phase2.py - Phase 2: DVCP + SAN 联合训练

加载 Phase 1 的 LoRA 权重，联合训练:
  1. 标准去噪损失 (MSE)
  2. DVCP 兼容性损失 (BCE对比学习)
  3. DVCP 生成惩罚 (对非法组合加重)
  4. SAN 工位自适应归一化

用法:
  accelerate launch --multi_gpu --num_processes 4 scripts/train_phase2.py \\
      --config configs/idgs.yaml --phase1_ckpt checkpoints/idgs/phase1_final
"""
import argparse
import os
import sys
import math
import logging
from pathlib import Path

import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.datasets.defect_dataset import DefectDataset
from src.models.dvcp_module import DVCPModule, DVCPConditioner
from src.models.san_module import SANWrapper
from scripts.train_phase1 import DefectGenerationDataset, make_prompt, tokenize_prompt

logger = get_logger(__name__, log_level="INFO")


def train(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]["phase2"]
    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]

    # Accelerator
    project_config = ProjectConfiguration(
        project_dir=cfg["output"]["checkpoint_dir"],
        logging_dir=cfg["output"]["log_dir"],
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation", 2),
        mixed_precision=train_cfg["mixed_precision"],
        log_with="tensorboard",
        project_config=project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

    # ── 加载模型 ──
    logger.info("Loading SD model + Phase 1 LoRA weights...")
    tokenizer = CLIPTokenizer.from_pretrained(
        model_cfg["sd_model_id"], subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_cfg["sd_model_id"], subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        model_cfg["sd_model_id"], subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_cfg["sd_model_id"], subfolder="unet"
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_cfg["sd_model_id"], subfolder="scheduler"
    )

    # 加载 Phase 1 LoRA
    if args.phase1_ckpt:
        logger.info(f"Loading Phase 1 LoRA from {args.phase1_ckpt}")
        unet = PeftModel.from_pretrained(unet, args.phase1_ckpt)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # ── 初始化 DVCP + SAN ──
    dvcp_module    = DVCPModule(defect_dim=64, cam_dim=64, hidden_dim=128)
    dvcp_cond      = DVCPConditioner(cam_dim=64, defect_dim=64, pose_dim=32, out_dim=768)
    san_wrapper    = SANWrapper(feature_dims=[4, 320, 640, 1280])

    # ── 数据集 ──
    logger.info("Loading dataset (NG + OK for Phase 2)...")
    base_ds = DefectDataset(
        root=data_cfg["root"],
        mode="all",          # Phase 2 用全部数据（OK+NG）
    )
    gen_ds = DefectGenerationDataset(
        base_ds, tokenizer, image_size=data_cfg["image_size"]
    )
    dataloader = DataLoader(
        gen_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"Dataset size: {len(gen_ds)} samples (OK+NG)")

    # ── 优化器 (LoRA + DVCP + SAN 联合优化) ──
    trainable_params = (
        list(filter(lambda p: p.requires_grad, unet.parameters()))
        + list(dvcp_module.parameters())
        + list(dvcp_cond.parameters())
        + list(san_wrapper.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=train_cfg["learning_rate"])

    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / train_cfg.get("gradient_accumulation", 2)
    )
    max_train_steps = train_cfg["epochs"] * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=200 * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    # ── Accelerator 准备 ──
    unet, dvcp_module, dvcp_cond, san_wrapper, optimizer, dataloader, lr_scheduler = (
        accelerator.prepare(
            unet, dvcp_module, dvcp_cond, san_wrapper,
            optimizer, dataloader, lr_scheduler
        )
    )
    text_encoder = text_encoder.to(accelerator.device)
    vae = vae.to(accelerator.device)

    weight_dtype = torch.float16 if train_cfg["mixed_precision"] == "fp16" else torch.float32
    vae.to(dtype=weight_dtype)
    text_encoder.to(dtype=weight_dtype)

    dvcp_loss_weight = train_cfg.get("dvcp_loss_weight", 0.5)

    # ── 训练循环 ──
    logger.info("Starting Phase 2 training (DVCP + SAN joint)...")

    for epoch in range(train_cfg["epochs"]):
        unet.train()
        dvcp_module.train()
        dvcp_cond.train()
        san_wrapper.train()

        epoch_denoise_loss = 0.0
        epoch_dvcp_loss    = 0.0
        epoch_total_loss   = 0.0

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                cam_idx    = batch["cam_idx"]
                defect_idx = batch["defect_idx"]
                pose_idx   = batch["pose_idx"]

                # ── VAE 编码 ──
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # ── SAN 对 latent 做工位自适应归一化 ──
                latents_san = san_wrapper.apply_san(latents.float(), cam_idx)

                # ── 加噪 ──
                noise = torch.randn_like(latents_san)
                bsz = latents_san.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents_san.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents_san, noise, timesteps)

                # ── 文本条件 + DVCP 条件注入 ──
                with torch.no_grad():
                    text_emb = text_encoder(batch["input_ids"])[0]

                # DVCP 条件 (仅对 NG 样本注入, OK 样本 defect_idx=-1)
                ng_mask = defect_idx >= 0
                if ng_mask.any():
                    # 将 OK 的 defect_idx 临时设为 0 避免 embedding 报错
                    safe_defect_idx = defect_idx.clone()
                    safe_defect_idx[~ng_mask] = 0

                    dvcp_emb = dvcp_cond(cam_idx, safe_defect_idx, pose_idx)  # (B, 1, 768)

                    # NG 样本: text_emb + dvcp_emb；OK 样本: 仅 text_emb
                    dvcp_emb_masked = dvcp_emb * ng_mask.float().unsqueeze(-1).unsqueeze(-1)
                    encoder_hidden_states = torch.cat([
                        text_emb.to(dtype=weight_dtype),
                        dvcp_emb_masked.to(dtype=weight_dtype)
                    ], dim=1)
                else:
                    # 全为 OK 的 batch，补零
                    zero_emb = torch.zeros(bsz, 1, 768, device=text_emb.device, dtype=weight_dtype)
                    encoder_hidden_states = torch.cat([text_emb.to(dtype=weight_dtype), zero_emb], dim=1)

                # ── UNet 去噪 ──
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                # ── Loss 1: 去噪 MSE (对 NG 样本使用 DVCP 加权) ──
                if ng_mask.any():
                    denoise_loss = dvcp_module.generation_penalty(
                        safe_defect_idx, cam_idx,
                        noise_pred.float(), noise.float()
                    )
                else:
                    denoise_loss = F.mse_loss(noise_pred.float(), noise.float())

                # ── Loss 2: DVCP 兼容性学习 (仅 NG 样本) ──
                if ng_mask.any():
                    ng_defect = defect_idx[ng_mask]
                    ng_cam    = cam_idx[ng_mask]
                    dvcp_loss = dvcp_module.dvcp_loss(ng_defect, ng_cam)
                else:
                    dvcp_loss = torch.tensor(0.0, device=accelerator.device)

                # ── 总损失 ──
                total_loss = denoise_loss + dvcp_loss_weight * dvcp_loss

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            epoch_denoise_loss += denoise_loss.detach().item()
            epoch_dvcp_loss    += dvcp_loss.detach().item()
            epoch_total_loss   += total_loss.detach().item()

        n = len(dataloader)
        logger.info(
            f"Epoch {epoch+1}/{train_cfg['epochs']} | "
            f"denoise={epoch_denoise_loss/n:.4f} | "
            f"dvcp={epoch_dvcp_loss/n:.4f} | "
            f"total={epoch_total_loss/n:.4f}"
        )

        # 保存
        if (epoch + 1) % train_cfg["save_every_n_epochs"] == 0:
            if accelerator.is_main_process:
                save_path = Path(cfg["output"]["checkpoint_dir"]) / f"phase2_epoch{epoch+1:03d}"
                save_path.mkdir(parents=True, exist_ok=True)
                accelerator.save_state(str(save_path))

                # 保存 DVCP 热力图数据
                matrix = accelerator.unwrap_model(dvcp_module).get_learned_matrix()
                import numpy as np
                np.save(str(save_path / "dvcp_matrix.npy"), matrix)
                logger.info(f"Checkpoint + DVCP matrix saved to {save_path}")

    # ── 最终保存 ──
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = Path(cfg["output"]["checkpoint_dir"]) / "phase2_final"
        final_path.mkdir(parents=True, exist_ok=True)

        unwrapped_unet = accelerator.unwrap_model(unet)
        unwrapped_unet.save_pretrained(str(final_path / "unet_lora"))

        torch.save(accelerator.unwrap_model(dvcp_module).state_dict(),
                    str(final_path / "dvcp_module.pt"))
        torch.save(accelerator.unwrap_model(dvcp_cond).state_dict(),
                    str(final_path / "dvcp_conditioner.pt"))
        torch.save(accelerator.unwrap_model(san_wrapper).state_dict(),
                    str(final_path / "san_wrapper.pt"))

        import numpy as np
        matrix = accelerator.unwrap_model(dvcp_module).get_learned_matrix()
        np.save(str(final_path / "dvcp_matrix_final.npy"), matrix)

        logger.info(f"All Phase 2 weights saved to {final_path}")

    accelerator.end_training()
    logger.info("Phase 2 training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      type=str, default="configs/idgs.yaml")
    parser.add_argument("--phase1_ckpt", type=str, default=None,
                        help="Path to Phase 1 LoRA checkpoint")
    args = parser.parse_args()
    train(args)

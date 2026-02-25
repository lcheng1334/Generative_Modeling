"""
train_phase1.py - SD v1.5 + LoRA 微调
在 4×RTX 3090 服务器上运行

用法:
  单卡测试:
    python scripts/train_phase1.py --config configs/idgs.yaml

  多卡训练 (4x3090):
    accelerate launch --multi_gpu --num_processes 4 scripts/train_phase1.py --config configs/idgs.yaml

  少样本模式 (每类50张):
    accelerate launch --multi_gpu --num_processes 4 scripts/train_phase1.py \\
        --config configs/idgs.yaml --max_per_class 50
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
from peft import LoraConfig, get_peft_model

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.datasets.defect_dataset import (
    DefectDataset, DEFECT_TYPES, POSES, CAM_IDS
)

logger = get_logger(__name__, log_level="INFO")


# ─────────────────────────────────────────────────────────
# 1. 条件文本构建
# ─────────────────────────────────────────────────────────
def make_prompt(cam: int, defect: str, pose: str) -> str:
    """
    把 (cam, defect, pose) 组合转成文本 prompt
    让 CLIP text encoder 理解条件

    例:
      "industrial inductor, cam1, p0, diffusion defect, bright field"
    """
    # 明场/暗场
    field = "dark field" if cam in [4, 6] else "bright field"
    # 分辨率提示
    res = "high resolution" if cam in [1, 2] else "medium resolution"

    prompt = (
        f"industrial inductor component, camera {cam}, {pose} orientation, "
        f"{defect.replace('_', ' ')} defect, {field}, {res}, "
        f"AOI inspection image"
    )
    return prompt


def make_ok_prompt(cam: int, pose: str) -> str:
    """OK 图的 prompt（用于训练时对比）"""
    field = "dark field" if cam in [4, 6] else "bright field"
    return (
        f"industrial inductor component, camera {cam}, {pose} orientation, "
        f"no defect, normal, {field}, AOI inspection image"
    )


def tokenize_prompt(tokenizer, prompt: str):
    return tokenizer(
        prompt,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids


# ─────────────────────────────────────────────────────────
# 2. 数据集包装（加 prompt）
# ─────────────────────────────────────────────────────────
class DefectGenerationDataset(torch.utils.data.Dataset):
    """
    在 DefectDataset 基础上加文本 prompt，用于 SD 训练
    """

    def __init__(self, base_dataset: DefectDataset, tokenizer, image_size=256):
        self.ds = base_dataset
        self.tokenizer = tokenizer
        self.image_size = image_size

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size),
                              interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds.samples[idx]

        # 图像
        img = Image.open(sample["path"]).convert("RGB")
        pixel_values = self.transform(img)

        # 文本 prompt
        if sample["label"] == 1:  # NG
            prompt = make_prompt(
                cam=sample["cam"],
                defect=sample["defect"],
                pose=sample["pose"],
            )
        else:  # OK
            prompt = make_ok_prompt(
                cam=sample["cam"],
                pose=sample["pose"],
            )

        input_ids = tokenize_prompt(self.tokenizer, prompt)[0]

        return {
            "pixel_values": pixel_values,
            "input_ids":    input_ids,
            "cam_idx":      torch.tensor(sample["cam_idx"],    dtype=torch.long),
            "defect_idx":   torch.tensor(sample["defect_idx"], dtype=torch.long),
            "pose_idx":     torch.tensor(sample["pose_idx"],   dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────
# 3. LoRA 配置
# ─────────────────────────────────────────────────────────
def setup_lora(unet, cfg):
    lora_config = LoraConfig(
        r=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    return unet


# ─────────────────────────────────────────────────────────
# 4. 主训练函数
# ─────────────────────────────────────────────────────────
def train(args):
    # 读配置
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_cfg = cfg["train"]["phase1"]
    data_cfg  = cfg["data"]
    model_cfg = cfg["model"]

    max_per_class = args.max_per_class or cfg["few_shot"]["max_per_class"]

    # Accelerator
    project_config = ProjectConfiguration(
        project_dir=cfg["output"]["checkpoint_dir"],
        logging_dir=cfg["output"]["log_dir"],
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=train_cfg["gradient_accumulation"],
        mixed_precision=train_cfg["mixed_precision"],
        log_with="tensorboard",
        project_config=project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"Accelerator state: {accelerator.state}")

    # ── 加载模型 ──
    logger.info(f"Loading SD model: {model_cfg['sd_model_id']}")
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

    # 冻结 VAE 和 text encoder，只训练 UNet LoRA
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # 注入 LoRA
    unet = setup_lora(unet, model_cfg)

    # ── 数据集 ──
    logger.info("Loading dataset...")
    base_ds = DefectDataset(
        root=data_cfg["root"],
        mode="ng",          # Phase 1 只用 NG 图微调
        max_per_class=max_per_class,
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
    logger.info(f"Dataset size: {len(gen_ds)} samples")

    # ── 优化器 ──
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=train_cfg["learning_rate"],
    )

    num_update_steps_per_epoch = math.ceil(
        len(dataloader) / train_cfg["gradient_accumulation"]
    )
    max_train_steps = train_cfg["epochs"] * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=train_cfg["lr_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=train_cfg["warmup_steps"] * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    # ── Accelerator 准备 ──
    unet, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, dataloader, lr_scheduler
    )
    text_encoder = text_encoder.to(accelerator.device)
    vae = vae.to(accelerator.device)

    if train_cfg["mixed_precision"] == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32
    vae.to(dtype=weight_dtype)
    text_encoder.to(dtype=weight_dtype)

    # ── 训练循环 ──
    logger.info("Starting Phase 1 training...")
    global_step = 0

    for epoch in range(train_cfg["epochs"]):
        unet.train()
        epoch_loss = 0.0

        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                # VAE 编码
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 加噪
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 文本条件
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # UNet 去噪预测
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states.to(dtype=weight_dtype),
                ).sample

                # MSE 去噪损失
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        unet.parameters(), train_cfg["max_grad_norm"]
                    )
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.detach().item()

            if accelerator.sync_gradients:
                global_step += 1

        avg_loss = epoch_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{train_cfg['epochs']} | loss={avg_loss:.4f} | lr={lr_scheduler.get_last_lr()[0]:.2e}")

        # 保存 checkpoint
        if (epoch + 1) % train_cfg["save_every_n_epochs"] == 0:
            if accelerator.is_main_process:
                save_path = Path(cfg["output"]["checkpoint_dir"]) / f"phase1_epoch{epoch+1:03d}"
                accelerator.save_state(str(save_path))
                logger.info(f"Checkpoint saved to {save_path}")

    # 最终保存 LoRA 权重
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_path = Path(cfg["output"]["checkpoint_dir"]) / "phase1_final"
        final_path.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(unet)
        unwrapped.save_pretrained(str(final_path))
        logger.info(f"Final LoRA weights saved to {final_path}")

    accelerator.end_training()
    logger.info("Phase 1 training complete!")


# ─────────────────────────────────────────────────────────
# 5. 入口
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        type=str, default="configs/idgs.yaml")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Few-shot limit per (cam, defect) group. Overrides config.")
    args = parser.parse_args()
    train(args)

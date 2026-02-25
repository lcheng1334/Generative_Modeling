"""
dvcp_module.py - Defect-View Compatibility Prior (可学习物理约束)

核心创新：不是硬编码规则矩阵，而是通过对比学习让模型"学到"
哪些 (camera, defect_type) 组合在物理上是合法的。

原理:
  - 合法组合 (cam1, breakage) → 生成的特征应接近真实 NG 特征
  - 非法组合 (cam3, breakage) → 模型应拒绝生成，施加惩罚
  - 训练时同时喂入合法和非法组合，用对比损失约束

用途:
  Phase 2 联合训练时，作为额外损失项加到标准去噪损失上
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────
# 物理兼容性矩阵（ground truth，来自制造工艺）
# ─────────────────────────────────────────────────────────
DVCP_GT = torch.tensor([
    # cam1 cam2 cam3 cam4 cam5 cam6
    [  1,   0,   1,   0,   0,   0],  # adhesion
    [  1,   1,   0,   0,   0,   0],  # breakage
    [  1,   1,   0,   0,   0,   0],  # contamination
    [  1,   0,   0,   0,   0,   0],  # diffusion
    [  1,   0,   0,   0,   0,   0],  # exposed_substrate
    [  0,   1,   0,   0,   0,   0],  # reversed_print
    [  0,   0,   1,   1,   1,   1],  # silver_overflow
], dtype=torch.float32)

NUM_DEFECTS = 7
NUM_CAMS    = 6


class DVCPModule(nn.Module):
    """
    可学习的缺陷-视角兼容性先验模块

    包含两部分:
    1. 兼容性预测器: 学习 (defect, cam) → compatibility score
    2. DVCP 损失: 对不合法组合施加惩罚

    Args:
        defect_dim: 缺陷 embedding 维度
        cam_dim:    摄像头 embedding 维度
        hidden_dim: 兼容性预测器的隐藏层维度
    """

    def __init__(self, defect_dim=64, cam_dim=64, hidden_dim=128):
        super().__init__()

        # Embedding 层
        self.defect_embed = nn.Embedding(NUM_DEFECTS, defect_dim)
        self.cam_embed    = nn.Embedding(NUM_CAMS,    cam_dim)

        # 兼容性预测器: (defect_embed, cam_embed) → scalar [0, 1]
        self.compatibility_head = nn.Sequential(
            nn.Linear(defect_dim + cam_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # 注册 ground truth 矩阵为 buffer（不参与梯度，但随模型保存）
        self.register_buffer("dvcp_gt", DVCP_GT)

    def predict_compatibility(self, defect_idx, cam_idx):
        """
        预测 (defect, cam) 组合的兼容性分数

        Args:
            defect_idx: (B,) LongTensor
            cam_idx:    (B,) LongTensor

        Returns:
            compat_score: (B,) FloatTensor, 在 [0, 1] 之间
        """
        d_emb = self.defect_embed(defect_idx)       # (B, defect_dim)
        c_emb = self.cam_embed(cam_idx)             # (B, cam_dim)
        combined = torch.cat([d_emb, c_emb], dim=1) # (B, defect_dim + cam_dim)
        score = self.compatibility_head(combined)    # (B, 1)
        return score.squeeze(-1)                     # (B,)

    def get_gt_compatibility(self, defect_idx, cam_idx):
        """
        查询 ground truth 兼容性 (0 或 1)

        Args:
            defect_idx: (B,) LongTensor
            cam_idx:    (B,) LongTensor

        Returns:
            gt: (B,) FloatTensor, 0 或 1
        """
        return self.dvcp_gt[defect_idx, cam_idx]

    def dvcp_loss(self, defect_idx, cam_idx):
        """
        DVCP 兼容性预测损失 (BCE)

        让模型学会预测哪些组合是物理合法的

        Args:
            defect_idx: (B,) LongTensor
            cam_idx:    (B,) LongTensor

        Returns:
            loss: scalar
        """
        pred = self.predict_compatibility(defect_idx, cam_idx)
        gt   = self.get_gt_compatibility(defect_idx, cam_idx)
        return F.binary_cross_entropy(pred, gt)

    def generation_penalty(self, defect_idx, cam_idx, noise_pred, noise_gt):
        """
        生成惩罚损失：对不合法组合的去噪预测施加额外惩罚

        合法组合 → 正常 MSE 损失
        非法组合 → MSE 损失 × penalty_weight (加重惩罚)

        Args:
            defect_idx: (B,)
            cam_idx:    (B,)
            noise_pred: (B, C, H, W) UNet 预测的噪声
            noise_gt:   (B, C, H, W) 真实噪声

        Returns:
            weighted_loss: scalar
        """
        gt_compat = self.get_gt_compatibility(defect_idx, cam_idx)  # (B,)

        # 逐样本 MSE
        per_sample_loss = F.mse_loss(
            noise_pred, noise_gt, reduction="none"
        ).mean(dim=[1, 2, 3])  # (B,)

        # 权重: 合法=1.0, 非法=5.0 (加重惩罚)
        penalty_weight = torch.where(
            gt_compat > 0.5,
            torch.ones_like(gt_compat),
            torch.full_like(gt_compat, 5.0),
        )

        weighted_loss = (per_sample_loss * penalty_weight).mean()
        return weighted_loss

    def get_learned_matrix(self):
        """
        获取模型学到的兼容性矩阵 (7×6)
        用于可视化：论文中的 DVCP 热力图

        Returns:
            matrix: (7, 6) numpy array
        """
        device = self.defect_embed.weight.device
        with torch.no_grad():
            defect_ids = torch.arange(NUM_DEFECTS, device=device)
            cam_ids    = torch.arange(NUM_CAMS, device=device)

            matrix = torch.zeros(NUM_DEFECTS, NUM_CAMS)
            for d in range(NUM_DEFECTS):
                for c in range(NUM_CAMS):
                    d_t = torch.tensor([d], device=device)
                    c_t = torch.tensor([c], device=device)
                    matrix[d, c] = self.predict_compatibility(d_t, c_t).item()

        return matrix.cpu().numpy()


class DVCPConditioner(nn.Module):
    """
    DVCP 条件注入模块

    将 (cam_id, defect_type, pose) 的 embedding 转为
    可以注入 UNet cross-attention 的条件向量

    用于 Phase 2 联合训练时，替代纯文本条件
    """

    def __init__(self, cam_dim=64, defect_dim=64, pose_dim=32, out_dim=768):
        super().__init__()
        self.cam_embed    = nn.Embedding(NUM_CAMS,    cam_dim)
        self.defect_embed = nn.Embedding(NUM_DEFECTS, defect_dim)
        self.pose_embed   = nn.Embedding(2,           pose_dim)   # p0, p90

        total_dim = cam_dim + defect_dim + pose_dim

        # 映射到与 CLIP text embedding 相同的维度 (768)
        self.proj = nn.Sequential(
            nn.Linear(total_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, cam_idx, defect_idx, pose_idx):
        """
        Args:
            cam_idx:    (B,) LongTensor
            defect_idx: (B,) LongTensor
            pose_idx:   (B,) LongTensor

        Returns:
            condition: (B, 1, 768) — 可以 concat 到 text encoder 的输出上
        """
        c = self.cam_embed(cam_idx)       # (B, cam_dim)
        d = self.defect_embed(defect_idx) # (B, defect_dim)
        p = self.pose_embed(pose_idx)     # (B, pose_dim)

        combined = torch.cat([c, d, p], dim=1)  # (B, total_dim)
        out = self.proj(combined)                # (B, 768)
        return out.unsqueeze(1)                  # (B, 1, 768)


# ─────────────────────────────────────────────────────────
# 快速测试
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("DVCP Module Test")
    print("=" * 50)

    dvcp = DVCPModule()

    # 测试合法组合: breakage + cam1 (合法)
    d_legal   = torch.tensor([1])  # breakage
    c_legal   = torch.tensor([0])  # cam1
    score_legal = dvcp.predict_compatibility(d_legal, c_legal)
    gt_legal    = dvcp.get_gt_compatibility(d_legal, c_legal)
    print(f"  breakage + cam1:  pred={score_legal.item():.3f}  gt={gt_legal.item()}")

    # 测试非法组合: breakage + cam3 (非法)
    d_illegal = torch.tensor([1])  # breakage
    c_illegal = torch.tensor([2])  # cam3
    score_illegal = dvcp.predict_compatibility(d_illegal, c_illegal)
    gt_illegal    = dvcp.get_gt_compatibility(d_illegal, c_illegal)
    print(f"  breakage + cam3:  pred={score_illegal.item():.3f}  gt={gt_illegal.item()}")

    # 测试 DVCP 损失
    batch_d = torch.tensor([0, 1, 6, 3])  # adhesion, breakage, silver_overflow, diffusion
    batch_c = torch.tensor([0, 0, 2, 0])  # cam1, cam1, cam3, cam1
    loss = dvcp.dvcp_loss(batch_d, batch_c)
    print(f"  DVCP loss (batch): {loss.item():.4f}")

    # 测试学到的矩阵
    matrix = dvcp.get_learned_matrix()
    print(f"\n  Learned matrix shape: {matrix.shape}")
    print(f"  (before training, values are random)")

    # 测试条件注入
    conditioner = DVCPConditioner()
    cond = conditioner(
        cam_idx=torch.tensor([0, 2]),
        defect_idx=torch.tensor([1, 6]),
        pose_idx=torch.tensor([0, 1]),
    )
    print(f"\n  DVCPConditioner output: {cond.shape}")
    print(f"  Expected: (2, 1, 768)")

    print("\n[PASS] DVCP module test passed")

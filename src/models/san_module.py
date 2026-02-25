"""
san_module.py - Station-Adaptive Normalization (工位自适应归一化)

目的:
  Cam1~6 的成像条件差异极大:
  - Cam1/2：明场, 1440x1080
  - Cam3/5：明场, 720x540
  - Cam4/6：暗场, 720x540

  SAN 通过为每工位学习一组归一化参数 (gamma, beta)，
  让同一个 UNet 能适应不同工位的成像风格。

实现:
  类似 AdaIN (Adaptive Instance Normalization)，
  但 gamma/beta 由 cam_id embedding 预测。
"""
import torch
import torch.nn as nn


NUM_CAMS = 6


class StationAdaptiveNorm(nn.Module):
    """
    工位自适应归一化

    给定一个特征图和 cam_id，输出经过工位特定归一化的特征图。

    Args:
        num_features: 输入特征图的通道数
        cam_dim:      摄像头 embedding 维度
    """

    def __init__(self, num_features, cam_dim=64):
        super().__init__()

        self.cam_embed = nn.Embedding(NUM_CAMS, cam_dim)

        # 预测 gamma (scale) 和 beta (shift)
        self.gamma_pred = nn.Sequential(
            nn.Linear(cam_dim, num_features),
        )
        self.beta_pred = nn.Sequential(
            nn.Linear(cam_dim, num_features),
        )

        # Instance Norm (不含学习参数，参数由 cam_id 决定)
        self.norm = nn.InstanceNorm2d(num_features, affine=False)

    def forward(self, x, cam_idx):
        """
        Args:
            x:       (B, C, H, W) 特征图
            cam_idx: (B,) LongTensor, 工位索引 0~5

        Returns:
            out: (B, C, H, W) 归一化后的特征图
        """
        cam_emb = self.cam_embed(cam_idx)              # (B, cam_dim)
        gamma   = self.gamma_pred(cam_emb).unsqueeze(-1).unsqueeze(-1) + 1.0  # (B, C, 1, 1)
        beta    = self.beta_pred(cam_emb).unsqueeze(-1).unsqueeze(-1)          # (B, C, 1, 1)

        normalized = self.norm(x)  # (B, C, H, W)
        out = normalized * gamma + beta
        return out


class SANWrapper(nn.Module):
    """
    SAN 封装模块

    用于在 UNet 的中间层注入工位自适应归一化。
    可以作为 hook 或显式调用。

    Args:
        feature_dims: 需要注入 SAN 的各层特征维度列表
        cam_dim:      摄像头 embedding 维度
    """

    def __init__(self, feature_dims=None, cam_dim=64):
        super().__init__()
        if feature_dims is None:
            # SD v1.5 UNet 的典型中间层维度
            feature_dims = [320, 640, 1280, 1280]

        self.san_layers = nn.ModuleDict({
            f"san_{dim}": StationAdaptiveNorm(dim, cam_dim)
            for dim in feature_dims
        })

    def apply_san(self, x, cam_idx):
        """
        根据特征图的通道数自动选择对应的 SAN 层

        Args:
            x:       (B, C, H, W)
            cam_idx: (B,)

        Returns:
            out: (B, C, H, W)
        """
        dim = x.shape[1]
        key = f"san_{dim}"
        if key in self.san_layers:
            return self.san_layers[key](x, cam_idx)
        return x  # 如果没有对应层，直接返回


# ─────────────────────────────────────────────────────────
# 快速测试
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("SAN Module Test")
    print("=" * 50)

    # 单层测试
    san = StationAdaptiveNorm(num_features=320)
    x = torch.randn(2, 320, 32, 32)
    cam_idx = torch.tensor([0, 3])  # cam1, cam4
    out = san(x, cam_idx)
    print(f"  StationAdaptiveNorm: input={x.shape} -> output={out.shape}")

    # 多层封装测试
    wrapper = SANWrapper()
    for dim in [320, 640, 1280]:
        x_test = torch.randn(2, dim, 16, 16)
        out_test = wrapper.apply_san(x_test, cam_idx)
        print(f"  SANWrapper dim={dim}: input={x_test.shape} -> output={out_test.shape}")

    # 不匹配的维度 → 直通
    x_pass = torch.randn(2, 256, 8, 8)
    out_pass = wrapper.apply_san(x_pass, cam_idx)
    print(f"  SANWrapper dim=256 (passthrough): input=output={out_pass.shape}")

    # 参数量统计
    total_params = sum(p.numel() for p in wrapper.parameters())
    print(f"\n  Total SAN parameters: {total_params:,}")

    print("\n[PASS] SAN module test passed")

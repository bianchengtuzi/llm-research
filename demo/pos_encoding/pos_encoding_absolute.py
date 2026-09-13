"""
绝对位置编码 (Absolute Positional Encoding)
========================================

自注意力本身是"无序"的（置换等变），需要给每个位置注入位置信息。

核心公式：
    x_i = TokenEmbed(w_i) + PosEmbed(i)

两种主流实现：
1. SinusoidalPE  —— 原始 Transformer 使用，固定公式，不可学习
2. LearnablePE   —— BERT / GPT-2 使用，每个位置一个可训练向量

运行：
    python demo/pos_encoding_absolute.py
"""

import torch
import torch.nn as nn


# ============================================================
# 输入模拟
# ============================================================

def make_demo_input(vocab_size=100, seq_len=8, d_model=16, seed=42):
    """构造一段可复现的假输入：token_id + token_embedding。"""
    g = torch.Generator().manual_seed(seed)
    token_ids = torch.randint(0, vocab_size, (seq_len,), generator=g)
    token_emb = nn.Embedding(vocab_size, d_model)
    x = token_emb(token_ids)          # (seq_len, d_model)
    print(f"[输入] token_ids:  {token_ids.tolist()}")
    print(f"[输入] token_emb:  shape={tuple(x.shape)}, dtype={x.dtype}")
    return x, token_ids


# ============================================================
# 1. 正弦余弦位置编码 (Sinusoidal PE, 原始 Transformer)
# ============================================================

class SinusoidalPE(nn.Module):
    """
    公式：
        PE(pos, 2i)   = sin(pos / 10000^(2i/d))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    - 低频维度变化慢（远距离敏感），高频维度变化快（近距离敏感）
    - 不学习，完全由公式生成
    - 性质：PE(pos+k) 可由 PE(pos) 线性旋转得到 → 注意力能学到相对位置
    """

    def __init__(self, d_model: int, max_len: int = 512, base: float = 10000.0):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)                # (max_len, 1)
        dim_idx = torch.arange(0, d_model, 2)                            # 偶数维度索引
        div_term = base ** (dim_idx / d_model)                           # 不同频率
        pe[:, 0::2] = torch.sin(position / div_term)                    # 偶数维度 → sin
        pe[:, 1::2] = torch.cos(position / div_term)                    # 奇数维度 → cos
        self.register_buffer("pe", pe)                                   # 不参与训练，跟随模型 to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (seq_len, d_model) 或 (batch, seq_len, d_model)"""
        seq_len = x.shape[-2]
        pos = self.pe[:seq_len]
        if x.dim() == 3:
            pos = pos.unsqueeze(0)                                      # (1, seq_len, d_model)
        return x + pos


# ============================================================
# 2. 可学习位置嵌入 (Learnable PE, BERT / GPT-2)
# ============================================================

class LearnablePE(nn.Module):
    """
    每个位置一个可训练向量：
        P ∈ R^(max_len × d_model)

    - 简单直接，表达能力强
    - 缺点：最大长度固定，超过训练长度无法外推
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)                       # 这就是所有要学的位置参数

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        positions = torch.arange(seq_len, device=x.device)
        pos = self.pe(positions)
        if x.dim() == 3:
            pos = pos.unsqueeze(0)
        return x + pos


# ============================================================
# 验证：正弦编码的线性旋转性质
# ============================================================

def verify_rotation_property(pe: SinusoidalPE, d_model: int, pos: int, k: int):
    """
    验证正弦编码的核心性质：PE(pos+k) 可以由 PE(pos) 经 -k 步旋转得到。

    即：PE(pos + k) = R(-k·θ) · PE(pos)

    这意味着注意力分数 (q·PE(m)) · (k·PE(n)) 的点积只依赖 (n-m)，
    因为 PE(m)^T PE(n) = PE(0)^T PE(n-m)。
    """
    from math import cos, sin

    pe_mat = pe.pe
    v1 = pe_mat[pos]
    v2 = pe_mat[pos + k]

    # 每对维度用不同频率：θ_j = base^(-2j/d)，j = 0, 1, ..., d/2 - 1
    rotated = torch.zeros_like(v1)
    for pair_idx in range(d_model // 2):
        theta = k * (10000.0 ** (-2 * pair_idx / d_model))
        c, s = cos(theta), sin(theta)
        x0 = v1[2 * pair_idx]
        x1 = v1[2 * pair_idx + 1]
        rotated[2 * pair_idx]     =  c * x0 + s * x1
        rotated[2 * pair_idx + 1] = -s * x0 + c * x1

    err = (rotated - v2).abs().max().item()
    print(f"  PE(pos={pos}+k={k})  vs PE(pos) 经 -k 旋转 误差: {err:.2e}  "
          f"{'✓ 通过' if err < 1e-5 else '✗ 未通过'}")
    return err < 1e-5


# ============================================================
# 可视化：不同维度的频率
# ============================================================

def visualize_frequencies(pe: SinusoidalPE, d_model: int, max_len: int = 100):
    """展示每个维度随位置变化的曲线 —— 低频 vs 高频一目了然。"""
    pe_mat = pe.pe[:max_len].T                                        # (d_model, max_len)
    print(f"\n[正弦编码频率分布]  d_model={d_model}")
    print(f"  维度 0 (sin, 最低频): {pe_mat[0, :8].tolist()}")
    print(f"  维度 1 (cos, 最低频): {pe_mat[1, :8].tolist()}")
    print(f"  维度 {d_model-2} (sin, 最高频): {pe_mat[d_model-2, :8].tolist()}")
    print(f"  维度 {d_model-1} (cos, 最高频): {pe_mat[d_model-1, :8].tolist()}")


# ============================================================
# 主函数
# ============================================================

def main():
    torch.manual_seed(42)
    d_model = 16
    seq_len = 8
    print("=" * 60)
    print("绝对位置编码 (Absolute Positional Encoding)")
    print("=" * 60)

    # 1) 构造输入
    print("\n--- 输入模拟 ---")
    x, _ = make_demo_input(seq_len=seq_len, d_model=d_model)

    # 2) 正弦余弦 PE
    print("\n--- Sinusoidal PE (原始 Transformer) ---")
    sin_pe = SinusoidalPE(d_model=d_model, max_len=64)
    out_sin = sin_pe(x)
    print(f"输出 shape: {tuple(out_sin.shape)}")
    print(f"示例：位置 0 vs 位置 1 的编码差范数 = "
          f"{(sin_pe.pe[0] - sin_pe.pe[1]).norm():.4f}")

    # 频率分布
    visualize_frequencies(sin_pe, d_model)

    # 验证旋转性质
    print("\n[旋转性质验证]")
    verify_rotation_property(sin_pe, d_model, pos=5, k=3)
    verify_rotation_property(sin_pe, d_model, pos=10, k=7)

    # 3) 可学习 PE
    print("\n--- Learnable PE (BERT / GPT-2) ---")
    learn_pe = LearnablePE(d_model=d_model, max_len=64)
    out_learn = learn_pe(x)
    print(f"输出 shape: {tuple(out_learn.shape)}")
    print(f"可训练参数量: {sum(p.numel() for p in learn_pe.parameters())} "
          f"= {64} × {d_model}")

    # 4) 两种 PE 的数值差异
    diff = (out_sin - out_learn).abs().mean().item()
    print(f"\n两种 PE 输出的平均绝对差: {diff:.4f}  (随机初始化可学习 PE 与固定正弦 PE 当然不同)")

    # 5) 置换等变演示
    print("\n--- 为什么自注意力需要位置编码？---")
    print("把输入顺序打乱：如果没有 PE，打乱前后注意力输出只是换位置，模型"
          "无法分辨 '猫追狗' 和 '狗追猫'。")
    idx = torch.tensor([3, 1, 7, 0, 5, 2, 6, 4])
    x_shuffled = x[idx]
    out_sin_shuf = sin_pe(x_shuffled)
    out_sin_orig_at_0 = sin_pe.pe[0]
    is_injected = (out_sin_shuf[0] != out_sin_orig_at_0).any().item()
    print(f"打乱后位置 0 的编码 与 原位置 0 的编码 不同？ {is_injected}  ✓")

    print("\n✓ 完成")


if __name__ == "__main__":
    main()

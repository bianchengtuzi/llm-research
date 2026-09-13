"""
相对位置编码 (Relative Positional Encoding)
============================================

不关心"我是第几个 token"，只关心"我和你相距多远"。
在注意力分数上加一个偏置项：

    e_ij = (q_i · k_j) / √d + b_{i-j}

典型实现：T5 的分桶相对位置偏置 (bucketed relative bias)

思路：
  - 近距离精确区分：距离 0, 1, 2, 3, ... 各自一个桶
  - 远距离对数分桶：距离 4~7 一个桶, 8~15 一个桶, ...
  - 这样无论多长序列，桶数都是有限的 → 可以预计算 bias 表

运行：
    python demo/pos_encoding_relative.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 输入模拟
# ============================================================

def make_demo_input(vocab_size=100, seq_len=6, d_model=8, seed=7):
    g = torch.Generator().manual_seed(seed)
    token_ids = torch.randint(0, vocab_size, (seq_len,), generator=g)
    emb = nn.Embedding(vocab_size, d_model)
    x = emb(token_ids)
    print(f"[输入] token_ids: {token_ids.tolist()}")
    print(f"[输入] shape: {tuple(x.shape)}")
    return x


# ============================================================
# 1. T5 风格：分桶相对位置 bias
# ============================================================

def _relative_position_bucket(
    relative_position: torch.Tensor,
    bidirectional: bool = True,
    num_buckets: int = 32,
    max_distance: int = 128,
) -> torch.Tensor:
    """
    将相对距离映射到桶索引。

    映射规则（bidirectional=True）：
      - 正数距离 (pos 向前看)  → 占一半桶
      - 负数距离 (pos 向后看)  → 占另一半桶
      - 近距离 (≤num_buckets/4) 每个距离一个桶
      - 远距离 对数分桶：4~7 → 同桶, 8~15 → 同桶, ...

    返回：与 relative_position 同 shape 的 bucket index 张量。
    """
    ret = torch.zeros_like(relative_position, dtype=torch.long)
    n = -relative_position                                       # k 在 q 前是正
    num_buckets //= 2                                            # 一半正一半负
    ret += (n > 0).long() * num_buckets                          # 正的加偏移
    n = n.abs()
    max_exact = num_buckets // 2                                 # 精确分桶的最大距离
    is_small = n < max_exact                                     # 近距离精确分桶
    # 远距离：对数缩放，均匀映射到剩余桶
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact)
        / torch.log(torch.tensor(max_distance / max_exact, dtype=torch.float))
        * (num_buckets - max_exact)
    ).long()
    val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
    ret += torch.where(is_small, n, val_if_large)
    return ret


class T5RelativeBias(nn.Module):
    """
    T5 的相对位置偏置。

    参数：
        num_buckets: 分桶总数（默认 32）
        max_distance: 对数分桶的最远距离（默认 128）
        num_heads: 每个注意力头有独立的 bias 参数

    用法：
        bias = T5RelativeBias(num_heads=4)     # (seq_len, seq_len) → (1, heads, seq, seq)
        scores = torch.matmul(q, k.transpose(-2, -1)) / sqrt_d
        scores = scores + bias(seq_len=8)
    """

    def __init__(self, num_heads: int, num_buckets: int = 32,
                 max_distance: int = 128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        # 每个桶每个头有一个可学习标量偏置
        self.rel_embedding = nn.Embedding(num_buckets, num_heads)

    def forward(self, seq_len: int) -> torch.Tensor:
        q_pos = torch.arange(seq_len).unsqueeze(1)                 # (seq, 1)
        k_pos = torch.arange(seq_len).unsqueeze(0)                 # (1, seq)
        rel_pos = q_pos - k_pos                                    # (seq, seq) 相对距离 i-j
        buckets = _relative_position_bucket(
            rel_pos, bidirectional=True,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )                                                          # (seq, seq)
        bias = self.rel_embedding(buckets)                         # (seq, seq, heads)
        bias = bias.permute(2, 0, 1).unsqueeze(0)                  # (1, heads, seq, seq)
        return bias


# ============================================================
# 2. 带相对 bias 的简化自注意力
# ============================================================

class SelfAttentionWithRelativeBias(nn.Module):
    """
    展示相对位置 bias 如何注入注意力分数。
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.rel_bias = T5RelativeBias(num_heads=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)                                      # (B, heads, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = self.head_dim ** -0.5
        scores = (q @ k.transpose(-2, -1)) * scale                 # (B, heads, T, T)
        scores = scores + self.rel_bias(T)                         # ← 核心：加相对位置偏置

        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(out), attn


# ============================================================
# 3. 验证：bias 对距离敏感
# ============================================================

def verify_bias_distance_sensitive(bias_module: T5RelativeBias, seq_len: int):
    """
    相对 bias 应该：距离越近 → bias 越大（更重要）；
                    距离越远 → bias 越小（或不同桶共享相同 bias）。
    """
    bias = bias_module(seq_len)                                   # (1, heads, seq, seq)
    head_0 = bias[0, 0]                                           # (seq, seq)
    print(f"\n[相对 bias 矩阵] head 0, shape={tuple(head_0.shape)}")
    print("行=query位置, 列=key位置, 数值越大=越被注意：")
    print(head_0.detach().numpy().round(2))

    # 同一 query 对不同距离 key 的 bias
    for q in range(seq_len):
        row = head_0[q]
        print(f"  query pos {q}:  bias 值 = {row.detach().numpy().round(2).tolist()}")


# ============================================================
# 4. 分桶映射可视化
# ============================================================

def visualize_bucketing(num_buckets=32, max_distance=16):
    """展示每个相对距离被分到哪个桶。"""
    dists = torch.arange(-max_distance, max_distance + 1)
    buckets = _relative_position_bucket(dists, num_buckets=num_buckets)
    pairs = list(zip(dists.tolist(), buckets.tolist()))
    print(f"\n[分桶映射]  num_buckets={num_buckets}:")
    print(f"  {'距离':>4} → 桶索引")
    for d, b in pairs:
        mark = " ← exact" if abs(d) < num_buckets // 4 else ""
        print(f"  {d:>4} → {b:>2}{mark}")


# ============================================================
# 主函数
# ============================================================

def main():
    torch.manual_seed(42)
    d_model = 16
    num_heads = 4
    seq_len = 8

    print("=" * 60)
    print("相对位置编码 (T5 分桶相对位置偏置)")
    print("=" * 60)

    # 1) 输入
    print("\n--- 输入 ---")
    x = make_demo_input(seq_len=seq_len, d_model=d_model)
    x_batch = x.unsqueeze(0)                                      # (1, seq, d_model)

    # 2) 分桶映射可视化
    visualize_bucketing(num_buckets=32, max_distance=16)

    # 3) 构造 bias 模块 & 展示 bias 矩阵
    print("\n--- T5RelativeBias ---")
    bias_mod = T5RelativeBias(num_heads=num_heads, num_buckets=32, max_distance=16)
    verify_bias_distance_sensitive(bias_mod, seq_len)

    # 4) 带 bias 的自注意力
    print("\n--- 带相对 bias 的自注意力 ---")
    sa = SelfAttentionWithRelativeBias(d_model=d_model, num_heads=num_heads)
    out, attn = sa(x_batch)
    print(f"输入  shape: {tuple(x_batch.shape)}")
    print(f"输出  shape: {tuple(out.shape)}")
    print(f"Attn  shape: {tuple(attn.shape)}  (B, heads, T, T)")

    # 5) 对比：打乱位置后 bias 跟着相对距离走
    print("\n--- 相对编码 vs 绝对编码的本质区别 ---")
    print("  绝对编码：bias[i,j] 依赖 i 和 j 的绝对值")
    print("  相对编码：bias[i,j] 只依赖 (i-j)，平移不改变 bias")
    print(f"验证：bias[0,2] == bias[3,5] (都是距离 2) ? "
          f"{(bias_mod(seq_len)[0, :, 0, 2] == bias_mod(seq_len)[0, :, 3, 5]).all().item()}")
    print(f"验证：bias[0,0] != bias[0,1] (距离不同) ? "
          f"{(bias_mod(seq_len)[0, :, 0, 0] != bias_mod(seq_len)[0, :, 0, 1]).any().item()}")

    print("\n✓ 完成")


if __name__ == "__main__":
    main()

"""
RoPE 旋转位置编码 (Rotary Positional Embedding)
===============================================

当前大模型主流位置编码方案：LLaMA / Qwen / Mistral / Gemma 均使用。

核心思想：
    不把位置向量加到 token embedding 上，而是在注意力计算时，
    对 Q 和 K 向量按位置做二维旋转。

    设位置 m 的 query 为 q_m，位置 n 的 key 为 k_n：

        q_m = R(m·θ) · q         k_n = R(n·θ) · k

    则它们的点积：

        q_m^T k_n = q^T R((n-m)θ) k

    结果只依赖相对位置 (n-m)，而与绝对位置 m、n 无关！

二维旋转矩阵：
    R(θ) = [[cosθ, -sinθ], [sinθ, cosθ]]

多维处理：把 head_dim 拆成很多二维子空间，每对维度用不同频率旋转。
频率公式：θ_i = base^(-2i/d)

运行：
    python demo/pos_encoding_rope.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 输入模拟
# ============================================================

def make_demo_input(d_model=8, num_heads=2, seq_len=6, batch=1, seed=42):
    """构造假的 Q、K、V 张量，形状 (B, heads, T, head_dim)。"""
    g = torch.Generator().manual_seed(seed)
    head_dim = d_model // num_heads
    Q = torch.randn(batch, num_heads, seq_len, head_dim, generator=g)
    K = torch.randn(batch, num_heads, seq_len, head_dim, generator=g)
    V = torch.randn(batch, num_heads, seq_len, head_dim, generator=g)
    print(f"[输入] Q={tuple(Q.shape)}, K={tuple(K.shape)}, V={tuple(V.shape)}")
    print(f"       num_heads={num_heads}, head_dim={head_dim}, seq_len={seq_len}")
    return Q, K, V


# ============================================================
# 1. 二维旋转矩阵
# ============================================================

def rotation_matrix_2d(theta: torch.Tensor) -> torch.Tensor:
    """
    由角度 θ 生成 2×2 旋转矩阵：
        R(θ) = [[cosθ, -sinθ], [sinθ, cosθ]]

    输入：theta (...)  —— 任意 shape 的角度张量
    输出：(..., 2, 2)  —— 每个角度对应一个旋转矩阵
    """
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.zeros(theta.shape + (2, 2), device=theta.device)
    R[..., 0, 0] = c
    R[..., 0, 1] = -s
    R[..., 1, 0] = s
    R[..., 1, 1] = c
    return R


def rotate_pairwise(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    对 x 的最后一维按 (2i, 2i+1) 两两配对旋转。

    x:     (..., d)      —— 最后一维必须是偶数
    angle: (..., d/2)    —— 每对维度的旋转角度

    输出同 shape，每对维度被各自角度旋转。
    """
    assert x.shape[-1] % 2 == 0
    x1 = x[..., 0::2]                                            # (..., d/2)
    x2 = x[..., 1::2]                                            # (..., d/2)
    c = torch.cos(angle)
    s = torch.sin(angle)
    out1 = c * x1 - s * x2
    out2 = s * x1 + c * x2
    # 交错拼回去
    out = torch.stack([out1, out2], dim=-1).flatten(-2)
    return out


# ============================================================
# 2. 预计算 RoPE 的 cos / sin 表
# ============================================================

class RoPECache(nn.Module):
    """
    预计算每个位置、每对维度的旋转角度 cos/sin。

    频率公式：θ_i = base^(-2i/d_head)    i = 0, 1, ..., d_head/2 - 1
    """

    def __init__(self, head_dim: int, max_seq_len: int = 4096,
                 base: float = 10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"
        self.head_dim = head_dim
        freq_idx = torch.arange(0, head_dim, 2, dtype=torch.float)  # 0, 2, 4, ...
        inv_freq = base ** (-freq_idx / head_dim)                   # (head_dim/2,)
        positions = torch.arange(max_seq_len, dtype=torch.float)   # (max_len,)
        # 每个 (pos, pair) 的角度 = pos * inv_freq[pair]
        angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)    # (max_len, head_dim/2)
        self.register_buffer("cos_cached", angles.cos())
        self.register_buffer("sin_cached", angles.sin())

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        x: (B, H, T, hd) 或 (T, hd)
        offset: KV cache 场景，已有 offset 个位置

        返回旋转后的 x（原地修改副本）。
        """
        T = x.shape[-2]
        cos = self.cos_cached[offset:offset + T]                    # (T, hd/2)
        sin = self.sin_cached[offset:offset + T]
        # 展开到 x 的维度
        while cos.dim() < x.dim() - 1:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        # cos/sin shape: (..., T, hd/2)  →  需要 broadcast 到 x
        cos = cos.unsqueeze(-1).expand(-1, -1, -1, 2).flatten(-2)   # (..., T, hd)
        sin = sin.unsqueeze(-1).expand(-1, -1, -1, 2).flatten(-2)

        # 用 cos/sin 两两配对旋转（更高效，等价于对每个 pair 做 R(θ)）
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        c1 = cos[..., 0::2]
        s1 = sin[..., 0::2]
        rotated = torch.empty_like(x)
        rotated[..., 0::2] = c1 * x1 - s1 * x2
        rotated[..., 1::2] = s1 * x1 + c1 * x2
        return rotated


# ============================================================
# 3. 关键性质验证：点积只依赖相对位置
# ============================================================

def verify_relative_dot_product(rope: RoPECache, head_dim: int):
    """
    验证 RoPE 核心性质：旋转后的点积只依赖相对位置。
    使用 rope 模块自身的 forward 来保证实现一致。
    """
    torch.manual_seed(123)
    q = torch.randn(1, 1, 1, head_dim)                           # (B, H, T, hd)
    k = torch.randn(1, 1, 1, head_dim)

    # rope 支持 offset 参数模拟不同位置
    q_rot_3 = rope(q, offset=3)                                  # q 放在位置 3
    k_rot_7 = rope(k, offset=7)                                  # k 放在位置 7
    dot1 = (q_rot_3 * k_rot_7).sum().item()

    q_rot_100 = rope(q, offset=100)                              # q 放在位置 100
    k_rot_104 = rope(k, offset=104)                              # k 放在位置 104
    dot2 = (q_rot_100 * k_rot_104).sum().item()

    err = abs(dot1 - dot2)
    print(f"\n[关键性质验证]  相对距离相同 → 点积相同")
    print(f"  q@pos=3  · k@pos=7   = {dot1:.6f}   (相对距离 +4)")
    print(f"  q@pos=100 · k@pos=104 = {dot2:.6f}   (相对距离 +4)")
    print(f"  误差: {err:.2e}   {'✓ RoPE 成立' if err < 1e-5 else '✗ 不成立'}")
    return err < 1e-5


def verify_different_distance_different_dot(rope: RoPECache, head_dim: int):
    """相对距离不同 → 点积应该不同（位置信息被注入）。"""
    torch.manual_seed(456)
    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    q_0 = rope(q, offset=0)
    k_1 = rope(k, offset=1)
    k_5 = rope(k, offset=5)
    dot_close = (q_0 * k_1).sum().item()
    dot_far = (q_0 * k_5).sum().item()

    diff = abs(dot_close - dot_far)
    print(f"  q@pos=0 · k@pos=1 (近) = {dot_close:.6f}")
    print(f"  q@pos=0 · k@pos=5 (远) = {dot_far:.6f}")
    print(f"  差值: {diff:.4f}   {'✓ 近远被区分' if diff > 1e-3 else '✗ 未有效区分'}")


# ============================================================
# 4. RoPE 频率分布
# ============================================================

def visualize_frequencies(rope: RoPECache):
    """低频维度变化慢，高频维度变化快。"""
    T = 10
    cos = rope.cos_cached[:T]                                    # (T, hd/2)
    print(f"\n[RoPE cos 表] head_dim={rope.head_dim} → {rope.head_dim // 2} 对维度, T={T}")
    print(f"  第 0 对 (最低频) cos 值: {cos[:, 0].tolist()}")
    print(f"  第 1 对 cos 值:          {cos[:, 1].tolist()}")
    last_idx = rope.head_dim // 2 - 1
    print(f"  第 {last_idx} 对 (最高频) cos 值: {cos[:, last_idx].tolist()}")


# ============================================================
# 5. 带 RoPE 的自注意力
# ============================================================

class SelfAttentionWithRoPE(nn.Module):
    """LLaMA 风格的自注意力。"""

    def __init__(self, d_model: int, num_heads: int,
                 max_seq_len: int = 4096):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RoPECache(self.head_dim, max_seq_len)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # 只对 Q、K 加 RoPE，V 不加
        q = self.rope(q, offset=offset)                            # 原地旋转
        k = self.rope(k, offset=offset)

        scale = self.head_dim ** -0.5
        scores = q @ k.transpose(-2, -1) * scale                   # (B, H, T, T)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.o_proj(out)


# ============================================================
# 主函数
# ============================================================

def main():
    torch.manual_seed(42)
    d_model = 16
    num_heads = 2
    head_dim = d_model // num_heads
    seq_len = 6

    print("=" * 60)
    print("RoPE 旋转位置编码 (Rotary Positional Embedding)")
    print("=" * 60)

    # 1) 输入
    print("\n--- 输入模拟 (Q, K, V) ---")
    Q, K, V = make_demo_input(d_model=d_model, num_heads=num_heads,
                              seq_len=seq_len, batch=1)

    # 2) RoPE 缓存
    print("\n--- RoPECache 预计算 ---")
    rope = RoPECache(head_dim=head_dim, max_seq_len=128)
    print(f"cos/sin 缓存 shape: {rope.cos_cached.shape}")     # (max_len, hd/2)
    visualize_frequencies(rope)

    # 3) 对 Q/K 应用 RoPE
    print("\n--- 对 Q, K 应用 RoPE ---")
    Q_rot = rope(Q)
    _ = rope(K)                                                  # 演示：K 同样被旋转
    print(f"旋转前 Q shape: {tuple(Q.shape)}, 旋转后: {tuple(Q_rot.shape)}")
    print(f"  Q[0,0,0,:4]  旋转前: {Q[0, 0, 0, :4].detach().numpy().round(3).tolist()}")
    print(f"  Q[0,0,0,:4]  旋转后: {Q_rot[0, 0, 0, :4].detach().numpy().round(3).tolist()}")
    print(f"  注意：位置 0 的 cos=1, sin=0 → 旋转 0° → Q 不变")

    # 4) 关键性质验证
    print("\n--- RoPE 核心性质验证 ---")
    verify_relative_dot_product(rope, head_dim)
    verify_different_distance_different_dot(rope, head_dim)

    # 5) 带 RoPE 的完整自注意力
    print("\n--- 带 RoPE 的完整自注意力 ---")
    x = torch.randn(1, seq_len, d_model)
    sa_rope = SelfAttentionWithRoPE(d_model=d_model, num_heads=num_heads,
                                    max_seq_len=128)
    out = sa_rope(x)
    print(f"输入  shape: {tuple(x.shape)}")
    print(f"输出  shape: {tuple(out.shape)}")
    print(f"参数量: q/k/v/o_proj + RoPE(无参数) = "
          f"{sum(p.numel() for n, p in sa_rope.named_parameters())}")

    # 6) KV cache 场景
    print("\n--- KV cache 场景演示 ---")
    x1 = torch.randn(1, 4, d_model)                              # 先输入 4 token
    out1 = sa_rope(x1, offset=0)
    x2 = torch.randn(1, 1, d_model)                              # 再输入第 5 个 token
    out2 = sa_rope(x2, offset=4)                                 # RoPE offset=4, 即位置 4
    print(f"第 1 步: 处理 4 token, offset=0 → out shape {tuple(out1.shape)}")
    print(f"第 2 步: 处理 1 token, offset=4 → out shape {tuple(out2.shape)}")
    print(f"  RoPE 支持 KV cache，每个 token 的旋转角度 = 它在原序列中的位置 × θ ✓")

    print("\n✓ 完成")


if __name__ == "__main__":
    main()

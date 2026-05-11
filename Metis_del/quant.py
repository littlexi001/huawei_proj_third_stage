
import torch


class QuantFunc:
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        return x.abs().max() + 1e-6
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        return x / s
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        return x * s


class WeightQuant(QuantFunc):
    @classmethod
    @torch.no_grad()
    def quant(cls, w, eps: float = 1e-6, bits = 1):
        
        abs_mean = w.abs().mean()
        abs_std  = w.abs().std()
        
        max_w = 2 * abs_std + eps
        q_range = max_w / (2 ** bits)
        w_quant = w / q_range
        
        w_quant = w_quant.round() / (2 ** bits)
        w_quant = w_quant.clamp(-1, 1) * abs_mean
    
        return w_quant

class Cast2Fp4e2m1(QuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        return x.abs().max() / 6 + 1e-6
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        
        xsign = x.sign()
        x = x.abs() / (s / 2)
        
        
        x -= (x - 4).relu_() / 2 + (x - 8).relu_() / 4
        x.round_()
        x += (x - 4).relu_() + (x - 6).relu_() * 2      
        
        return x * xsign / 2
    
class Cast2Fp4e2m1Random(QuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        return x.abs().max() / 6 + 1e-6
    
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x:torch.Tensor, s: torch.Tensor):
        xsign = x.sign()
        x = x.abs() / (s / 2)
        
        x -= (x - 4).relu_() / 2 + (x - 8).relu_() / 4
        x += torch.rand_like(x) - 0.5
        x.round_()
        x += (x - 4).relu_() + (x - 6).relu_() * 2      
        return x * xsign / 2
        # return out * xsign

class Cast2Fp6e3m2(QuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        return x.abs().max() / 625 + 1e-7
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        x1 = (x / s).clamp(-625, 625).abs()
        x1 = (x1 ** (1 / 4)).to(torch.float8_e5m2).to(torch.float32)
        x1 = x1 ** 4

        return torch.sign(x) * x1

class Cast2Fp8e4m3(QuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        return x.abs().max() / 448 + 1e-6
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        return (x / s).to(dtype=torch.float8_e4m3fn).to(dtype=torch.float32)


class Cast2Fp32(QuantFunc):
    pass

class BlockQuantFunc(QuantFunc):
    block_shape = (1, 16)
                
    @classmethod
    @torch.no_grad()
    def _reshape(cls, x: torch.Tensor, s: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        s = s.view(rows // brows, 1, cols // bcols, 1)
        x = x.view(rows // brows, brows, cols // bcols, bcols)
        return x, s
    

class Cast2MXFp4e2m1Block(BlockQuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        assert(rows % brows == 0 and cols % bcols == 0)
        
        x = x.abs() \
             .view(rows // brows, brows, cols // bcols, bcols) \
             .amax(dim=(1, 3), keepdim=True) \
             .view(rows // brows, cols // bcols) \
             / 6 + 1e-9
        
        return x
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1Random.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        s = s.sign() * (2 ** ((s + 1e-127).log2().clamp_(-127, 127).round_()))
        
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1Random.rquant(x, s).view(xshape)
    

class Cast2MXFp4e2m1BlockNOSR(BlockQuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        assert(rows % brows == 0 and cols % bcols == 0)
        
        x = x.abs() \
             .view(rows // brows, brows, cols // bcols, bcols) \
             .amax(dim=(1, 3), keepdim=True) \
             .view(rows // brows, cols // bcols) \
             / 6 + 1e-9
        
        return x
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        s = s.sign() * (2 ** ((s + 1e-127).log2().clamp_(-127, 127).round_()))
        
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1.rquant(x, s).view(xshape)

class Cast2NVFp4e2m1Block(BlockQuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        assert(rows % brows == 0 and cols % bcols == 0)
        
        x = x.abs() \
             .view(rows // brows, brows, cols // bcols, bcols) \
             .amax(dim=(1, 3), keepdim=True) \
             .view(rows // brows, cols // bcols) \
             / 6 + 1e-9
        
        return x
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        # smax = s.abs().max()        
        # # print("Origin", s.max(), s.shape)
        # s /= smax / 448
        # # print("Before to",s.max())
        # s = s.to(dtype=torch.float8_e4m3fn).to(dtype=torch.float32) + 1e-10
        # # print("After to",s.max())
        # s *= smax / 448
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1Random.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        smax = s.abs().max()
        s /= smax / 448
        s = s.to(dtype=torch.float8_e4m3fn).to(dtype=torch.float32)
        s *= smax / 448
        
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1Random.rquant(x, s).view(xshape)
    
class Cast2NVFp4e2m1BlockNOSR(BlockQuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        assert(rows % brows == 0 and cols % bcols == 0)
        
        x = x.abs() \
             .view(rows // brows, brows, cols // bcols, bcols) \
             .amax(dim=(1, 3), keepdim=True) \
             .view(rows // brows, cols // bcols) \
             / 6 + 1e-9
        
        return x
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        # smax = s.abs().max()
        # s /= smax / 448
        # s = s.to(dtype=torch.float8_e4m3fn).to(dtype=torch.float32)
        # s *= smax / 448
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        smax = s.abs().max()
        s /= smax / 448
        s = s.to(dtype=torch.float8_e4m3fn).to(dtype=x.dtype)
        s *= smax / 448
        
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1.rquant(x, s).view(xshape)


class Cast2Fp4e2m1Block(BlockQuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        assert(rows % brows == 0 and cols % bcols == 0)
        
        x = x.abs() \
             .view(rows // brows, brows, cols // bcols, bcols) \
             .amax(dim=(1, 3), keepdim=True) \
             .view(rows // brows, cols // bcols) \
             / 6 + 1e-9
        
        return x
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1Random.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp4e2m1Random.rquant(x, s).view(xshape)
    
class Cast2Fp6e3m2Block(BlockQuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        x = x.view(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        assert(rows % brows == 0 and cols % bcols == 0)
        
        x = x.abs() \
             .view(rows // brows, brows, cols // bcols, bcols) \
             .amax(dim=(1, 3), keepdim=True) \
             .view(rows // brows, cols // bcols) \
             / 625 + 1e-7
        
        return x.to(dtype=torch.float16).to(dtype=torch.float32)
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp6e3m2.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp6e3m2.rquant(x, s).view(xshape)


class Cast2Fp8e4m3Block(BlockQuantFunc):
    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor):
        x = x.view(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        assert(rows % brows == 0 and cols % bcols == 0)
        
        x = x.abs() \
             .view(rows // brows, brows, cols // bcols, bcols) \
             .amax(dim=(1, 3), keepdim=True) \
             .view(rows // brows, cols // bcols) \
             / 448 + 1e-7
        
        return x.to(dtype=torch.float16).to(dtype=torch.float32)
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp8e4m3.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._reshape(x, s)
        return Cast2Fp8e4m3.rquant(x, s).view(xshape)

@torch.no_grad()
def cast_2_fp32(x):
    return x


# --------------------------------LNS开始-----------------------------------
class Cast2LNS4_QDQ(BlockQuantFunc):
    """
    LNS4 (4-bit) fake-quantization (QDQ style):
      - Linear-domain stochastic rounding onto {0} ∪ {±2^(k-bias), k=1..7}
      - Block-wise scaling controlled by *global* BlockQuantFunc.block_shape
      - rquant() casts s to FP8 e4m3 and back before multiplying
    """
    # 固定默认偏置（可随时在外部改：Cast2LNS4_QDQ.default_bias = 2/3/...）
    default_bias = 3

    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor, bias: int | None = None):
        """
        返回按块（或全局）计算的 scale 网格：
          s_block = amax_block / 2^(7 - bias) + eps
        说明：
          - 把归一化 |x|/s 后的可表示上界对齐到 2^(7-bias)
          - block_size 由 BlockQuantFunc.block_shape 控制
        """
        if bias is None:
            bias = cls.default_bias

        # 把输入摊成 [rows, cols]，按最后一维分块
        x2 = x.reshape(-1, x.shape[-1])
        rows, cols = x2.shape
        brows, bcols = BlockQuantFunc.block_shape    # 全局控制 block_size / global scaling
        assert (rows % brows == 0) and (cols % bcols == 0), "block_shape must divide input shape"

        # 每块取绝对值最大值（absmax）
        amax = (
            x2.abs()
              .view(rows // brows, brows, cols // bcols, bcols)
              .amax(dim=(1, 3), keepdim=False)       # [rows//brows, cols//bcols]
        )

        # 上界对齐：LNS4 非零最大幂是 2^(7-bias)
        max_nonzero = 2.0 ** (7 - bias)
        s = (amax / max_nonzero + 1e-9).to(torch.float32)   # FP32 网格
        return s

    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor, bias: int | None = None):
        """
        量化（QDQ 风格，返回仍为浮点）：
          1) 归一化 a = |x|/s
          2) 在线性域随机舍入到最近两幂（或 {0, 2^(1-bias)}），并按距离确定上抬概率
          3) 乘回符号，得到假量化浮点 q（仍 FP32）
        """
        if bias is None:
            bias = cls.default_bias

        xshape = x.shape
        x4, s4 = BlockQuantFunc._reshape(x, s)  # 把 s 按块广播到元素位
        xsign  = x4.sign()
        a      = (x4.abs() / s4).clamp(min=0.0)  # 归一化幅值，非负

        # LNS4 网格（归一化域）
        min_nz = 2.0 ** (1 - bias)   # 最小非零幂
        max_nz = 2.0 ** (7 - bias)   # 最大非零幂

        qmag = torch.zeros_like(a)

        # 区间一：0 < a < min_nz → 在 {0, min_nz} 之间随机
        mask = (a > 0) & (a < min_nz)
        if mask.any():
            p = (a[mask] / min_nz).clamp(0.0, 1.0)                       # 线性距离概率
            qmag[mask] = torch.where(torch.rand_like(p) < p, min_nz, 0.) # 随机择近

        # 区间二：min_nz ≤ a < max_nz → 在相邻幂 {rL, rU=2*rL} 之间随机
        mask = (a >= min_nz) & (a < max_nz)
        if mask.any():
            a2 = a[mask]
            k  = torch.floor(torch.log2(a2))       # 左侧幂指数
            rL = torch.pow(2.0, k)                 # 左端幂
            rU = rL * 2.0                          # 右端幂
            p  = ((a2 - rL) / (rU - rL)).clamp(0.0, 1.0)  # 线性插值概率
            qmag[mask] = torch.where(torch.rand_like(p) < p, rU, rL)

        # 区间三：a ≥ max_nz → 饱和到最大幂
        qmag = torch.where(a >= max_nz, torch.full_like(a, max_nz), qmag)

        q = (xsign * qmag).view(xshape)  # 假量化后的“浮点”值（仍在归一化域）
        return q

    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor, bias: int | None = None):
        """
        反量化（回实数域）：
          - 把 s cast 到 FP8 e4m3 再回 FP32（模拟低精读回）
          - 乘回：x_hat = x * s_fp8
          - 返回 FP32（供后续 GEMM/SVD 等浮点算子使用）
        """
        if bias is None:
            bias = cls.default_bias

        xshape = x.shape
        x4, s4 = BlockQuantFunc._reshape(x, s)
        s_fp8  = s4.to(torch.float8_e4m3fn).to(torch.float32)
        return (x4 * s_fp8).view(xshape)






class Cast2LNS4_nsr(BlockQuantFunc):
    """
    LNS4 (4-bit) fake-quantization (QDQ style):
      - Linear-domain ROUND-TO-NEAREST onto {0} ∪ {±2^(k-bias), k=1..7}
      - Block-wise scaling controlled by *global* BlockQuantFunc.block_shape
      - rquant() casts s to FP8 e4m3 and back before multiplying
    """
    # 固定默认偏置（可在外部随时改：Cast2LNS4_nsr.default_bias = 2/3/...）
    default_bias = 3

    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor, bias: int | None = None):
        """
        返回按块（或全局）计算的 scale 网格：
          s_block = amax_block / 2^(7 - bias) + eps
        说明：
          - 归一化 |x|/s 后的可表示上界对齐到 2^(7-bias)
          - block_size 由 BlockQuantFunc.block_shape 控制
        """
        if bias is None:
            bias = cls.default_bias

        # 摊成二维，按最后一维分块
        x2 = x.reshape(-1, x.shape[-1])
        rows, cols = x2.shape
        brows, bcols = BlockQuantFunc.block_shape
        assert (rows % brows == 0) and (cols % bcols == 0), "block_shape must divide input"

        # 每块 absmax
        amax = (
            x2.abs()
              .view(rows // brows, brows, cols // bcols, bcols)
              .amax(dim=(1, 3), keepdim=False)   # [rows//brows, cols//bcols]
        )

        # 上界对齐到 2^(7-bias)
        max_nonzero = 2.0 ** (7 - bias)
        s = (amax / max_nonzero + 1e-9).to(torch.float32)
        return s

    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor, bias: int | None = None):
        """
        量化（QDQ 风格，返回仍为浮点）：
          1) 归一化 a = |x|/s
          2) 在线性域“就近四舍五入”到最近的幂（或 {0, 2^(1-bias)}）
          3) 乘回符号，得到假量化浮点 q（仍 FP32）
        """
        if bias is None:
            bias = cls.default_bias

        xshape = x.shape
        x4, s4 = BlockQuantFunc._reshape(x, s)  # 把 s 按块广播到元素位
        xsign  = x4.sign()
        a      = (x4.abs() / s4).clamp(min=0.0)  # 归一化幅值

        # LNS4 网格（归一化域）
        min_nz = 2.0 ** (1 - bias)   # 最小非零幂
        max_nz = 2.0 ** (7 - bias)   # 最大非零幂

        # qmag：就近“四舍五入”到网格点
        qmag = torch.zeros_like(a)

        # 区间一：0 < a < min_nz   → 与 {0, min_nz} 比较，阈值为 min_nz/2
        mask1 = (a > 0) & (a < min_nz)
        if mask1.any():
            mid = 0.5 * min_nz
            qmag[mask1] = torch.where(a[mask1] >= mid, torch.full_like(a[mask1], min_nz), torch.zeros_like(a[mask1]))

        # 区间二：min_nz ≤ a < max_nz → 找相邻幂 rL 与 rU=2*rL，阈值为 (rL + rU)/2 = 1.5*rL
        mask2 = (a >= min_nz) & (a < max_nz)
        if mask2.any():
            a2 = a[mask2]
            k  = torch.floor(torch.log2(a2))
            rL = torch.pow(2.0, k)          # 左端幂
            rU = rL * 2.0                   # 右端幂
            mid = 0.5 * (rL + rU)           # 线性域中点
            qmag[mask2] = torch.where(a2 >= mid, rU, rL)

        # 区间三：a >= max_nz → 饱和到最大幂
        qmag = torch.where(a >= max_nz, torch.full_like(a, max_nz), qmag)

        q = (xsign * qmag).view(xshape)  # 假量化后的“浮点”值（仍在归一化域）
        return q

    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor, bias: int | None = None):
        """
        反量化（回实数域）：
          - 把 s cast 到 FP8 e4m3 再回 FP32（模拟低精读回）
          - 乘回：x_hat = x * s_fp8
          - 返回 FP32（供后续 GEMM/SVD 等浮点算子使用）
        """
        if bias is None:
            bias = cls.default_bias

        xshape = x.shape
        x4, s4 = BlockQuantFunc._reshape(x, s)
        s_fp8  = s4.to(torch.float8_e4m3fn).to(torch.float32)
        return (x4 * s_fp8).view(xshape)





class Cast2LNS4SRFP32(BlockQuantFunc):
    """
    LNS4 (4-bit) fake-quantization (QDQ style):
      - Linear-domain STOCHASTIC rounding onto {0} ∪ {±2^(k-bias), k=1..7}
      - Block-wise scaling controlled by *global* BlockQuantFunc.block_shape
      - rquant() multiplies FP32 scale directly (NO FP8 e4m3 roundtrip)
    """
    # 默认偏置，可外部修改：Cast2LNS4_QDQ.default_bias = 2/3/...
    default_bias = 0

    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor, bias: int | None = None):
        """
        计算每块（或全局）的 scale 网格：
          s_block = amax_block / 2^(7 - bias) + eps
        将归一化 |x|/s 的“最大非零”对齐到 2^(7-bias)。
        block_size 由全局 BlockQuantFunc.block_shape 控制。
        """
        if bias is None:
            bias = cls.default_bias

        # 摊成二维，按最后一维分块
        x2 = x.reshape(-1, x.shape[-1])
        rows, cols = x2.shape
        brows, bcols = BlockQuantFunc.block_shape
        assert (rows % brows == 0) and (cols % bcols == 0), "block_shape must divide input"

        # 每块 absmax → [rows//brows, cols//bcols]
        amax = (
            x2.abs()
              .view(rows // brows, brows, cols // bcols, bcols)
              .amax(dim=(1, 3), keepdim=False)
        )

        # 上界对齐：LNS4 非零最大幂 2^(7-bias)
        max_nonzero = 2.0 ** (7 - bias)
        s = (amax / max_nonzero + 1e-9).to(torch.float32)
        return s

    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor, bias: int | None = None):
        """
        量化（QDQ 风格，返回仍为浮点）：
          1) 归一化 a = |x|/s
          2) 在线性域“随机舍入”到最近两幂（或 {0, 2^(1-bias)}）
             - 小于最小非零：在 {0, min_nz} 间以 p=a/min_nz 概率上抬
             - 中间区间：在 {rL, rU=2*rL} 间以 p=(a-rL)/(rU-rL) 概率上抬
             - 超上界：饱和到 max_nz
          3) 乘回符号，得到假量化浮点 q（仍 FP32，归一化域）
        """
        if bias is None:
            bias = cls.default_bias

        xshape = x.shape
        x4, s4 = BlockQuantFunc._reshape(x, s)  # s 广播到元素位
        xsign  = x4.sign()
        a      = (x4.abs() / s4).clamp(min=0.0)

        # LNS4 网格（归一化域）
        min_nz = 2.0 ** (1 - bias)   # 最小非零幂
        max_nz = 2.0 ** (7 - bias)   # 最大非零幂

        qmag = torch.zeros_like(a)

        # 1) 0 < a < min_nz → 在 {0, min_nz} 间，p_up = a/min_nz
        mask = (a > 0) & (a < min_nz)
        if mask.any():
            p = (a[mask] / min_nz).clamp(0.0, 1.0)
            qmag[mask] = torch.where(torch.rand_like(p) < p, min_nz, 0.0)

        # 2) min_nz ≤ a < max_nz → 在 {rL, rU=2*rL} 间，p_up = (a-rL)/(rU-rL)
        mask = (a >= min_nz) & (a < max_nz)
        if mask.any():
            a2 = a[mask]
            k  = torch.floor(torch.log2(a2))
            rL = torch.pow(2.0, k)
            rU = rL * 2.0
            p  = ((a2 - rL) / (rU - rL)).clamp(0.0, 1.0)
            qmag[mask] = torch.where(torch.rand_like(p) < p, rU, rL)

        # 3) a ≥ max_nz → 饱和到 max_nz
        qmag = torch.where(a >= max_nz, torch.full_like(a, max_nz), qmag)

        q = (xsign * qmag).view(xshape)  # 假量化后的“浮点值”（仍在归一化域）
        return q

    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor, bias: int | None = None):
        """
        反量化（回实数域）：
          - 直接使用 FP32 的 s：x_hat = x * s
          - 不再进行 FP8 e4m3 roundtrip（更贴近“FP32 scale 存取”的配置）
        """
        if bias is None:
            bias = cls.default_bias

        xshape = x.shape
        x4, s4 = BlockQuantFunc._reshape(x, s)
        return (x4 * s4).view(xshape)



class Cast2LNS4NSRFP32(BlockQuantFunc):
    """
    LNS4 (4-bit) fake-quantization (QDQ style):
      - Linear-domain ROUND-TO-NEAREST onto {0} ∪ {±2^(k-bias), k=1..7}
      - Block-wise scaling controlled by *global* BlockQuantFunc.block_shape
      - rquant() multiplies FP32 scale directly (NO FP8 e4m3 roundtrip)
    """
    # 默认偏置，可外部随时改：Cast2LNS4_QDQ.default_bias = 0/2/3/...
    default_bias = 3

    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor, bias: int | None = None):
        """
        返回按块（或全局）计算的 scale 网格：
          s_block = amax_block / 2^(7 - bias) + eps
        说明：
          - 归一化 |x|/s 后的可表示上界对齐到 2^(7-bias)
          - block_size 由 BlockQuantFunc.block_shape 控制（可用于全局 / per-block）
        """
        if bias is None:
            bias = cls.default_bias

        # 摊成二维，按最后一维分块
        x2 = x.reshape(-1, x.shape[-1])
        rows, cols = x2.shape
        brows, bcols = BlockQuantFunc.block_shape
        assert (rows % brows == 0) and (cols % bcols == 0), "block_shape must divide input"

        # 每块 absmax → [rows//brows, cols//bcols]
        amax = (
            x2.abs()
              .view(rows // brows, brows, cols // bcols, bcols)
              .amax(dim=(1, 3), keepdim=False)
        )

        # 上界对齐：LNS4 非零最大幂 2^(7-bias)
        max_nonzero = 2.0 ** (7 - bias)
        s = (amax / max_nonzero + 1e-9).to(torch.float32)
        return s

    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor, bias: int | None = None):
        """
        量化（QDQ 风格，返回仍为浮点）：
          1) 归一化 a = |x|/s
          2) 在线性域“就近四舍五入”到幂网格
             - 0 < a < min_nz      : 与 {0, min_nz} 比较，阈值 min_nz/2
             - min_nz ≤ a < max_nz : 与 {rL, rU=2*rL} 比较，阈值 (rL+rU)/2
             - a ≥ max_nz          : 饱和到 max_nz
          3) 乘回符号，得到假量化浮点 q（仍 FP32，归一化域）
        """
        if bias is None:
            bias = cls.default_bias

        xshape = x.shape
        x4, s4 = BlockQuantFunc._reshape(x, s)  # s 广播到元素位
        xsign  = x4.sign()
        a      = (x4.abs() / s4).clamp(min=0.0)

        # LNS4 网格（归一化域）
        min_nz = 2.0 ** (1 - bias)   # 最小非零：2^(1-bias)
        max_nz = 2.0 ** (7 - bias)   # 最大非零：2^(7-bias)

        qmag = torch.zeros_like(a)

        # 1) 0 < a < min_nz：与 {0, min_nz} 比较，阈值 min_nz/2
        mask1 = (a > 0) & (a < min_nz)
        if mask1.any():
            mid = 0.5 * min_nz
            qmag[mask1] = torch.where(a[mask1] >= mid,
                                      torch.full_like(a[mask1], min_nz),
                                      torch.zeros_like(a[mask1]))

        # 2) min_nz ≤ a < max_nz：找相邻幂 rL 与 rU=2*rL，阈值 (rL + rU)/2 = 1.5*rL
        mask2 = (a >= min_nz) & (a < max_nz)
        if mask2.any():
            a2 = a[mask2]
            k  = torch.floor(torch.log2(a2))
            rL = torch.pow(2.0, k)                # 左端幂
            rU = rL * 2.0                         # 右端幂
            mid = 0.5 * (rL + rU)                 # 线性域中点
            qmag[mask2] = torch.where(a2 >= mid, rU, rL)

        # 3) a ≥ max_nz：饱和
        qmag = torch.where(a >= max_nz, torch.full_like(a, max_nz), qmag)

        q = (xsign * qmag).view(xshape)  # 假量化“浮点”（仍在归一化域）
        return q

    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor, bias: int | None = None):
        """
        反量化（回实数域）：
          - 直接使用 FP32 的 s：x_hat = x * s
          - 不进行 FP8 e4m3 roundtrip（更接近“scale 以 FP32 存取”的配置）
        """
        if bias is None:
            bias = cls.default_bias

        xshape = x.shape
        x4, s4 = BlockQuantFunc._reshape(x, s)
        return (x4 * s4).view(xshape)



class Cast2LNS4_SR_GlobalFP8(QuantFunc):
    """
    LNS4 (4-bit) fake-quant (QDQ) with per-tensor/global scaling:
      - Linear-domain STOCHASTIC rounding onto {0} ∪ {±2^(k-bias), k=1..7}
      - ONE scale for the whole tensor (inherits QuantFunc)
      - rquant() casts the scale s to FP8 e4m3 then back to FP32 before multiplying
    """
    # 默认偏置（可外部一行改：Cast2LNS4_QDQ_GlobalFP8.default_bias = 0/2/3/...）
    default_bias = 3

    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor, bias: int | None = None):
        """
        Per-tensor/global scale:
          s = abs(x).max() / 2^(7 - bias) + eps
        返回标量（或 0-d 张量，能广播到整张量）
        """
        if bias is None:
            bias = cls.default_bias
        amax = x.abs().max()
        max_nonzero = 2.0 ** (7 - bias)
        s = (amax / max_nonzero + 1e-9).to(torch.float32)
        return s

    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor, bias: int | None = None):
        """
        QDQ 量化（返回仍为浮点）：
          1) a = |x| / s
          2) 在线性域进行“随机舍入”到幂网格
             - 0 < a < min_nz      : 在 {0, min_nz} 间以上抬概率 p=a/min_nz 抽签
             - min_nz ≤ a < max_nz : 在 {rL, rU=2*rL} 间以上抬概率 p=(a-rL)/(rU-rL) 抽签
             - a ≥ max_nz          : 饱和到 max_nz
          3) 乘回符号，得到假量化浮点 q（仍在归一化域）
        """
        if bias is None:
            bias = cls.default_bias

        xsign = x.sign()
        a = (x.abs() / s).clamp(min=0.0)

        # LNS4 幂网格（归一化域）
        min_nz = 2.0 ** (1 - bias)   # 最小非零：2^(1-bias)
        max_nz = 2.0 ** (7 - bias)   # 最大非零：2^(7-bias)

        qmag = torch.zeros_like(a)

        # 1) 0 < a < min_nz → 在 {0, min_nz} 之间，p_up = a / min_nz
        mask = (a > 0) & (a < min_nz)
        if mask.any():
            p = (a[mask] / min_nz).clamp(0.0, 1.0)
            qmag[mask] = torch.where(torch.rand_like(p) < p, min_nz, 0.0)

        # 2) min_nz ≤ a < max_nz → 在 {rL, rU=2*rL} 之间，p_up = (a-rL)/(rU-rL)
        mask = (a >= min_nz) & (a < max_nz)
        if mask.any():
            a2 = a[mask]
            k  = torch.floor(torch.log2(a2))   # 左端幂指数
            rL = torch.pow(2.0, k)            # 左端幂
            rU = rL * 2.0                     # 右端幂
            p  = ((a2 - rL) / (rU - rL)).clamp(0.0, 1.0)
            qmag[mask] = torch.where(torch.rand_like(p) < p, rU, rL)

        # 3) a ≥ max_nz → 饱和到最大幂
        qmag = torch.where(a >= max_nz, torch.full_like(a, max_nz), qmag)

        # 乘回符号（仍是“假量化浮点”，尚未乘回 s）
        q = xsign * qmag
        return q

    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor, bias: int | None = None):
        """
        反量化（回实数域）：
          - 先把 s cast 到 FP8 e4m3 再回 FP32（模拟 scale 的低精读回）
          - 再乘回：x_hat = x * s_fp8
        """
        if bias is None:
            bias = cls.default_bias

        s_fp8 = s.to(torch.float8_e4m3fn).to(torch.float32)
        return x * s_fp8



class Cast2LNS4_NSR_GlobalFP8(QuantFunc):
    """
    LNS4 (4-bit) fake-quant (QDQ) with per-tensor/global scaling:
      - Linear-domain ROUND-TO-NEAREST onto {0} ∪ {±2^(k-bias), k=1..7}
      - ONE scale for the whole tensor (inherits QuantFunc)
      - rquant() casts the scale s to FP8 e4m3 then back to FP32 before multiplying
    """
    # 默认偏置，可外部一行改：Cast2LNS4_QDQ_GlobalFP8.default_bias = 0/2/3/...
    default_bias = 3

    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor, bias: int | None = None):
        """
        Per-tensor/global scale:
          s = abs(x).max() / 2^(7 - bias) + eps
        返回标量（或 0-d 张量，能广播到整张量）
        """
        if bias is None:
            bias = cls.default_bias
        amax = x.abs().max()
        max_nonzero = 2.0 ** (7 - bias)
        s = (amax / max_nonzero + 1e-9).to(torch.float32)
        return s

    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor, bias: int | None = None):
        """
        QDQ 量化（返回仍为浮点）：
          1) a = |x| / s
          2) 在线性域“就近四舍五入”到幂网格
             - 0 < a < min_nz      : 与 {0, min_nz} 比较，阈值 min_nz/2
             - min_nz ≤ a < max_nz : 与 {rL, rU=2*rL} 比较，阈值 (rL+rU)/2
             - a ≥ max_nz          : 饱和到 max_nz
          3) 乘回符号，得到假量化浮点 q（仍在归一化域）
        """
        if bias is None:
            bias = cls.default_bias

        xsign = x.sign()
        a = (x.abs() / s).clamp(min=0.0)

        # LNS4 幂网格（归一化域）
        min_nz = 2.0 ** (1 - bias)   # 最小非零：2^(1-bias)
        max_nz = 2.0 ** (7 - bias)   # 最大非零：2^(7-bias)

        qmag = torch.zeros_like(a)

        # 1) 0 < a < min_nz：与 {0, min_nz} 比较，阈值 min_nz/2
        mask1 = (a > 0) & (a < min_nz)
        if mask1.any():
            mid = 0.5 * min_nz
            qmag[mask1] = torch.where(
                a[mask1] >= mid,
                torch.full_like(a[mask1], min_nz),
                torch.zeros_like(a[mask1])
            )

        # 2) min_nz ≤ a < max_nz：找相邻幂 rL 与 rU=2*rL，阈值 (rL + rU)/2
        mask2 = (a >= min_nz) & (a < max_nz)
        if mask2.any():
            a2 = a[mask2]
            k  = torch.floor(torch.log2(a2))   # 左端幂指数
            rL = torch.pow(2.0, k)            # 左端幂
            rU = rL * 2.0                     # 右端幂
            mid = 0.5 * (rL + rU)             # 线性域中点
            qmag[mask2] = torch.where(a2 >= mid, rU, rL)

        # 3) a ≥ max_nz → 饱和到最大幂
        qmag = torch.where(a >= max_nz, torch.full_like(a, max_nz), qmag)

        # 乘回符号（仍是“假量化浮点”，尚未乘回 s）
        q = xsign * qmag
        return q

    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor, bias: int | None = None):
        """
        反量化（回实数域）：
          - 先把 s cast 到 FP8 e4m3 再回 FP32（模拟 scale 的低精读回）
          - 再乘回：x_hat = x * s_fp8
        """
        if bias is None:
            bias = cls.default_bias

        s_fp8 = s.to(torch.float8_e4m3fn).to(torch.float32)
        return x * s_fp8



class Cast2LNS4_QDQ_GlobalFP32(QuantFunc):
    """
    LNS4 (4-bit) fake-quant (QDQ) with per-tensor/global scaling:
      - Linear-domain STOCHASTIC rounding onto {0} ∪ {±2^(k-bias), k=1..7}
      - ONE scale for the whole tensor (inherits QuantFunc ➜ per-tensor scaling)
      - rquant() multiplies FP32 scale directly (NO FP8 e4m3 roundtrip)
    """
    # 默认偏置；可在外部一行修改：Cast2LNS4_QDQ_GlobalFP32.default_bias = 0/2/3/...
    default_bias = 0

    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor, bias: int | None = None):
        """
        Per-tensor/global scale:
          s = abs(x).max() / 2^(7 - bias) + eps
        返回标量（或 0-d 张量），可广播到整张量。
        """
        if bias is None:
            bias = cls.default_bias
        amax = x.abs().max()
        max_nonzero = 2.0 ** (7 - bias)
        s = (amax / max_nonzero + 1e-9).to(torch.float32)
        return s  # ★ per-tensor：整个张量一个 s

    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor, bias: int | None = None):
        """
        QDQ 量化（返回仍为浮点）：
          1) a = |x| / s
          2) 在线性域随机舍入到幂网格（无偏抽签）
             - 0 < a < min_nz      : 在 {0, min_nz} 间，p_up = a/min_nz
             - min_nz ≤ a < max_nz : 在 {rL, rU=2*rL} 间，p_up = (a-rL)/(rU-rL)
             - a ≥ max_nz          : 饱和到 max_nz
          3) 乘回符号，得到假量化浮点 q（仍在归一化域）
        """
        if bias is None:
            bias = cls.default_bias

        xsign = x.sign()
        a = (x.abs() / s).clamp(min=0.0)

        # LNS4 网格（归一化域）
        min_nz = 2.0 ** (1 - bias)   # 最小非零：2^(1-bias)
        max_nz = 2.0 ** (7 - bias)   # 最大非零：2^(7-bias)

        qmag = torch.zeros_like(a)

        # 1) 0 < a < min_nz → {0, min_nz}，p_up = a/min_nz
        mask = (a > 0) & (a < min_nz)
        if mask.any():
            p = (a[mask] / min_nz).clamp(0.0, 1.0)
            qmag[mask] = torch.where(torch.rand_like(p) < p, min_nz, 0.0)

        # 2) min_nz ≤ a < max_nz → {rL, rU=2*rL}，p_up = (a-rL)/(rU-rL)
        mask = (a >= min_nz) & (a < max_nz)
        if mask.any():
            a2 = a[mask]
            k  = torch.floor(torch.log2(a2))   # 左端幂指数
            rL = torch.pow(2.0, k)            # 左端幂
            rU = rL * 2.0                     # 右端幂
            p  = ((a2 - rL) / (rU - rL)).clamp(0.0, 1.0)
            qmag[mask] = torch.where(torch.rand_like(p) < p, rU, rL)

        # 3) a ≥ max_nz → 饱和
        qmag = torch.where(a >= max_nz, torch.full_like(a, max_nz), qmag)

        q = xsign * qmag  # 假量化浮点（仍在归一化域）
        return q

    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor, bias: int | None = None):
        """
        反量化（回实数域）：
          - 直接使用 FP32 的 scale：x_hat = x * s
          - 不进行 FP8 e4m3 roundtrip
        """
        if bias is None:
            bias = cls.default_bias
        return x * s



class Cast2LNS4_QDQ_GlobalFP32_RTN(QuantFunc):
    """
    LNS4 (4-bit) fake-quant (QDQ) with per-tensor/global scaling:
      - Linear-domain ROUND-TO-NEAREST onto {0} ∪ {±2^(k-bias), k=1..7}
      - ONE scale for the whole tensor (inherits QuantFunc ⇒ per-tensor scaling)
      - rquant() multiplies FP32 scale directly (NO FP8 e4m3 roundtrip)
    """
    # 默认偏置；可在外部一行修改：Cast2LNS4_QDQ_GlobalFP32_RTN.default_bias = 0/2/3/...
    default_bias = 0

    @classmethod
    @torch.no_grad()
    def get_scalar(cls, x: torch.Tensor, bias: int | None = None):
        """
        Per-tensor/global scale:
          s = abs(x).max() / 2^(7 - bias) + eps
        返回标量（或 0-d 张量），可广播到整张量。
        """
        if bias is None:
            bias = cls.default_bias
        amax = x.abs().max()
        max_nonzero = 2.0 ** (7 - bias)
        s = (amax / max_nonzero + 1e-9).to(torch.float32)
        return s  # ★ per-tensor：整个张量一个 s

    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor, bias: int | None = None):
        """
        QDQ 量化（返回仍为浮点）：
          1) a = |x| / s
          2) 在线性域“就近四舍五入”到幂网格（确定性）
             - 0 < a < min_nz      : 与 {0, min_nz} 比较，阈值 min_nz/2
             - min_nz ≤ a < max_nz : 与 {rL, rU=2*rL} 比较，阈值 (rL+rU)/2
             - a ≥ max_nz          : 饱和到 max_nz
          3) 乘回符号，得到假量化浮点 q（仍在归一化域）
        """
        if bias is None:
            bias = cls.default_bias

        xsign = x.sign()
        a = (x.abs() / s).clamp(min=0.0)

        # LNS4 幂网格（归一化域）
        min_nz = 2.0 ** (1 - bias)   # 最小非零：2^(1-bias)
        max_nz = 2.0 ** (7 - bias)   # 最大非零：2^(7-bias)

        qmag = torch.zeros_like(a)

        # 1) 0 < a < min_nz：与 {0, min_nz} 比较，阈值 min_nz/2
        mask1 = (a > 0) & (a < min_nz)
        if mask1.any():
            mid = 0.5 * min_nz
            qmag[mask1] = torch.where(
                a[mask1] >= mid,
                torch.full_like(a[mask1], min_nz),
                torch.zeros_like(a[mask1])
            )

        # 2) min_nz ≤ a < max_nz：找相邻幂 rL 与 rU=2*rL，阈值 (rL + rU)/2
        mask2 = (a >= min_nz) & (a < max_nz)
        if mask2.any():
            a2 = a[mask2]
            k  = torch.floor(torch.log2(a2))   # 左端幂指数
            rL = torch.pow(2.0, k)            # 左端幂
            rU = rL * 2.0                     # 右端幂
            mid = 0.5 * (rL + rU)             # 线性域中点
            qmag[mask2] = torch.where(a2 >= mid, rU, rL)

        # 3) a ≥ max_nz → 饱和到最大幂
        qmag = torch.where(a >= max_nz, torch.full_like(a, max_nz), qmag)

        # 乘回符号（仍是“假量化浮点”，尚未乘回 s）
        q = xsign * qmag
        return q

    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor, bias: int | None = None):
        """
        反量化（回实数域）：
          - 直接使用 FP32 的 s：x_hat = x * s
          - 不进行 FP8 e4m3 roundtrip
        """
        if bias is None:
            bias = cls.default_bias
        return x * s

# --------------------------------LNS结束-----------------------------------







quant_func = {
    "fp4e2m1": Cast2Fp4e2m1,
    "nvfp4e2m1b": Cast2NVFp4e2m1Block,
    "mxfp4e2m1b": Cast2MXFp4e2m1Block,    
    "mxfp4e2m1bnosr": Cast2MXFp4e2m1BlockNOSR,
    "nvfp4e2m1bnosr": Cast2NVFp4e2m1BlockNOSR,
    "fp6e3m2": Cast2Fp6e3m2,
    "fp6e3m2b": Cast2Fp6e3m2Block,
    "fp8e4m3": Cast2Fp8e4m3,
    "fp8e4m3b": Cast2Fp8e4m3Block,
    "fp32": Cast2Fp32,
    "1p58bit": WeightQuant,
    "lns4sre4m3": Cast2LNS4_QDQ,
    "lns4nsre4m3": Cast2LNS4_nsr,
    "lns4srfp32": Cast2LNS4SRFP32,
    "lns4nsrfp32": Cast2LNS4NSRFP32,
    "lns4srgfp8": Cast2LNS4_SR_GlobalFP8,
    "lns4nsrgfp8": Cast2LNS4_NSR_GlobalFP8,
    "lns4srgfp32": Cast2LNS4_QDQ_GlobalFP32,
    "lns4nsrgfp32": Cast2LNS4_QDQ_GlobalFP32_RTN,

}



if __name__ == "__main__":
    x = torch.randn([1, 16])
    print(x)
    s = Cast2Fp4e2m1Block.get_scalar(x)
    qx = Cast2Fp4e2m1Block.quant(x, s)
    qx = Cast2Fp4e2m1Block.rquant(qx, s)

    print(qx)
    # print(qx / s)
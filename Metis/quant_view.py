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
    def _view(cls, x: torch.Tensor, s: torch.Tensor):
        x = x.view(-1, x.shape[-1])
        rows = x.shape[0]
        cols = x.shape[1]
        
        brows = BlockQuantFunc.block_shape[0]
        bcols = BlockQuantFunc.block_shape[1]
        
        s = s.view(rows // brows, 1, cols // bcols, 1)
        x = x.view(rows // brows, brows, cols // bcols, bcols)
        return x, s
    

class Cast2Fp4e2m1Block(BlockQuantFunc):
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
             / 6 + 1e-9
        
        return x
    
    @classmethod
    @torch.no_grad()
    def quant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._view(x, s)
        return Cast2Fp4e2m1Random.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._view(x, s)
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
        x, s = BlockQuantFunc._view(x, s)
        return Cast2Fp6e3m2.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._view(x, s)
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
        x, s = BlockQuantFunc._view(x, s)
        return Cast2Fp8e4m3.quant(x, s).view(xshape)
    
    @classmethod
    @torch.no_grad()
    def rquant(cls, x: torch.Tensor, s: torch.Tensor):
        xshape = x.shape
        x, s = BlockQuantFunc._view(x, s)
        return Cast2Fp8e4m3.rquant(x, s).view(xshape)

@torch.no_grad()
def cast_2_fp32(x):
    return x




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





quant_func = {
    "fp4e2m1": Cast2Fp4e2m1,
    "fp4e2m1b": Cast2Fp4e2m1Block,
    "fp6e3m2": Cast2Fp6e3m2,
    "fp6e3m2b": Cast2Fp6e3m2Block,
    "fp8e4m3": Cast2Fp8e4m3,
    "fp8e4m3b": Cast2Fp8e4m3Block,
    "fp32": Cast2Fp32,
    "1p58bit": WeightQuant,
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
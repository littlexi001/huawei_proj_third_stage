# 专利：一种基于二维切片的低精度乘法加速硬件架构


## 一、 技术问题背景
在大语言模型（LLM）的算力加速领域，量化技术（如 INT4/FP4）是提升计算效率、降低显存占用的核心手段。然而，现有的量化方案主要采用 Block-wise（块级）均匀量化，通常以行或列为单位共享缩放系数（Scale）。
在复杂的神经网络计算逻辑中，激活值矩阵 $X$ 具有双重身份：
1. 前向传播$XW^T$： $X$ 作为输入，通常需要按行（Row-wise）进行量化，以便与权重矩阵 $W$ 进行高效的算子乘法。
2. 反向传播$\Delta W = X^\top \cdot G_{out}$，$\Delta X=G_{out}^T \cdot W$： $X$,$W$ 和 $G_{out}$ 均需参与梯度计算，且在前向、反向过程中，$X,W,G_{out}$分别以原矩阵和转置形式参与了两次运算，逻辑上需要按不同行列分别进行量化。

这种对维度依赖的量化方式导致了严重的效率瓶颈：
- 显存冗余： 由于无法预知反向传播所需的量化参数，系统必须在显存（HBM）中完整保留一份$X$的 16-bit 高精度副本。
- 计算浪费： 同一个矩阵$X$在前向和反向过程中需要被重复执行两次性质不同的量化操作。
- 带宽瓶颈： 即使使用了低精度算子，由于$X$需要以 16-bit 形式在显存与计算核心（SM）之间频繁搬运，导致“低精度计算”无法转化为“低精度存储与低精度传输”。

## 二、 现有方案
当前的硬件量化电路（如 Tensor Core 内部的量化逻辑）通常集成在乘法器前端，其工作逻辑如下：
1. 加载与暂存： 从显存读取 16-bit 数据至寄存器或 SRAM。
2. 一维量化： 根据算子需求，对数据流进行一维处理（行方向或列方向），计算该行 / 列的最大值并生成 Scale。
3. 计算： 执行低精度乘法，返回 16-bit 累加结果。
局限性： 这种设计是“维度敏感”的。一旦按行量化，其结果就无法直接用于需要按列量化的反向传播场景。这迫使软件栈在底层实现时，必须依赖 16-bit 高精度数据流作为“万能基准”，导致“低精度计算”无法转化为“低精度存储与低精度传输”。

## 三、 本发明提出的方案
本发明提出一种基于 Tile（二维切片）的自适应量化电路架构及软硬结合的数据流策略。其核心思想是将量化粒度从“一维向量”细化为“二维切片（Tile）”，实现量化结果在不同维度运算下的硬件级复用。
1. 二维 Tile 量化电路设计
重新设计乘法器前端的量化逻辑模块。不再支持长向量的行/列量化，而是强制以 $n \times n$（如 $8 \times 8$或 $16 \times 16$）的 Tile 为独立量化单元。每个 Tile 共享一个 Scale 因子。
- 对等性设计： 由于 Tile 是方阵，无论矩阵是处于原始状态还是转置状态，Tile 内部的数据分布特性保持不变。
- 硬件读取逻辑： 改造乘法器内部的索引电路。当从不同方向（左乘或右乘）读取该 Tile 时，电路能自动根据当前算子方向，动态映射 Tile 内部的低精度数据与对应的 Scale。

2. “一次量化，终身低精度”数据流
- 前向量化持久化： 在前向计算阶段，激活值$X$经过 Tile 量化器后，直接以 FP4（含Scale）的形式写回 HBM，不再保留 16-bit 副本。
- 反向数据复用： 梯度计算阶段，直接从 HBM 读取量化后的 FP4 Tile 数据。硬件量化电路通过识别 Tile 描述符，自动将 Scale 应用于转置后的访存流，无需重新计算量化过程。

3. 核心技术优势（硬件实现的必要性）
- 突破 HBM 带宽壁垒： 实现了真正意义上的全链路低精度传输。$X$ 的传输位宽从 16-bit 降低至 4-bit，带宽理论收益提升至 4倍。
- 寄存器与 SRAM 节省： 硬件直接处理 Tile 级别的 Scale，无需在软件层维护复杂的量化表，大幅降低了计算核心内部的临时存储压力。
- 端到端时延降低： 消除了反向传播时重复量化的计算开销，同时通过“Tile 化”规避了传统量化方法在维度变换时的同步气泡。

## 四、实施例：线性层 Tile 量化
为了清晰展示这一软硬结合的专利方案，我们将以一个典型的 Transformer 线性层（Linear Layer）为实施例。
初始状态：
- 权重$W\in\R^{d\times d}$： 已预先量化并存储在 HBM 中。
- 输入$X\in\R^{N\times d}$： 上一层传来的高精度（BF16/FP16）张量。
- 硬件支持： 具有 Tile 量化能力的量化电路（Quantization Unit）和支持 Tile 访存的乘法器阵列。
1. 前向计算中的 Tile 级在线量化（On-the-fly Quantization）：当执行前向传播 $Y = XW$ 时，SM 从 HBM 读取 $X$ 的高精度数据，电路以$n \times n$（如$8\times 8$）为窗口，实时计算该 Tile 的缩放因子 $S_{tile}$， 产生低精度 Tile 数据 $X_{q4}$。
2. “一次量化”的持久化存储：硬件将量化后的 $X_{q4}$ 及其关联的低精度 Scale $S_{tile}$ 直接写回 HBM，作为后续反向传播的输入。原始的高精度 $X$ 直接从显存中丢弃，不再占用 16-bit 空间的显存 Buffer。
3. 前向乘法执行：乘法器阵列读取量化后的 $X_{q4}$ 和预量化的 $W_{q4}$。乘法器根据 $S_{tile}$ 对计算结果进行反量化，并输出高精度（BF16）的中间结果 $Y$。
4. 反向传播启动：计算输入梯度 $\nabla X = G \cdot W^\top$ 或权重梯度 $\nabla W = X^\top \cdot G$ 时，从 HBM 加载步骤 2 存储的 $X_{q4}$（4-bit）和对应的 $S_{tile}$，以及反向传回的高精度梯度 $G$。
5. 硬件层面的“维度无关”映射（Tile-Reindexing）：由于 $X$ 在计算权重梯度 $\nabla W$ 时是以转置形式 $X^\top$ 出现的，硬件调度器识别到当前是转置逻辑，直接将 Tile 内的索引进行行列互换（Transpose within Tile）。
6. 梯度产出与写回：乘法器在低精度下完成 $X_{q4}^\top \cdot G$ 的局部累加。硬件累加器将不同 Tile 的部分和进行汇总，生成高精度的梯度结果 $\nabla W$ 或 $\nabla X$。

## 五、专利效果建模：单层线性层（Linear Layer）对比

### 一、HBM 占用
HBM 节省收益：在 $N \gg d$ 的情况下，通过将$X$的持久化精度从 16-bit 降至 4-bit，峰值占用减少了 37.5%。
1. Baseline 方案
- 前向：读取 16-bit $X$ ($2Nd$)，读取 16-bit $W$ ($2d^2$)。在 SM 内部量化后执行计算。由于反向需重新量化，$X$ 必须以 16-bit 持久化保留。
- 反向：保留的 16-bit $X$ ($2Nd$) + 16-bit $W$ 主副本 ($2d^2$) + 输入的后一层 grad output 16-bit$G$ ($2Nd$)。
- 计算产出：计算出给前一层的 grad input 16-bit $\nabla X$ ($2Nd$) 以及 16-bit $\nabla W$ ($2d^2$)。
- 最终 Baseline 峰值占用：约 $4Nd + 4d^2$ Bytes（主要涉及$X$、$G$、$\nabla X$三个激活值相关矩阵，以及$W$和 $\nabla W$两个权重相关矩阵，其中 grad input 可以在本层计算后覆盖后一层的 grad output）。
2. 本专利方案
- 前向：初始保存 16-bit $X$和$W$，但在量化为 4-bit Tile 后，立即释放原有的 $X$ 16-bit 空间，并保存一份 $W$的 4-bit 副本（$0.5d^2$）。
- 反向：持久化保留的 4-bit$X$ ($0.5Nd$) + 16-bit $W$ ($2d^2$) + 输入的 16-bit grad output$G$($2Nd$)。
- 计算产出：16-bit grad input $\nabla X$($2Nd$) + 16-bit $\nabla W$($2d^2$)。
- 峰值占用：约 $2.5Nd + 4.5d^2$ Bytes。

### 二、时延
单层线性计算整体时延缩减 27.95%。
1. Baseline 方案
$Total\_T = 2(T_{q\_X} + T_{q\_W} + T_{q\_G}) + T_{comp\_fwd} + T_{comp\_bwd} + \frac{6Nd + 4d^2}{BW}$
  - $T_{mem\_fwd}=\frac{2Nd+2d^2}{BW}$：前向从HBM读取 16-bit 的 $X$ 和 $W$ 的时延
  - $T_{q\_X}$：前向过程中对 $X$ 进行行量化的计算开销。
  - $T_{q\_W}$：对权重 $W$ 进行量化的开销。
  - $T_{comp\_fwd}$：前向矩阵乘法计算时延。
  - $T_{mem\_X\_bwd}=\frac{4Nd+2d^2}{BW}$：反向阶段从 HBM 读取 16-bit $X$ , $W$ 和 $G_{out}$ 的访存时延。
  - $T_{q\_X}$：反向过程中因维度变化，对$X$重新进行列量化的开销。
  - $T_{q\_G}$：对输出梯度$G$进行量化的开销，因计算$\nabla X$和$\nabla W$时还需分别重新量化W和G。
  - $T_{comp\_bwd}$：反向矩阵乘法计算时延（包含计算$\nabla X$和$\nabla W$）。

2. 本专利方案
$Total\_T = T_{q\_X} + T_{q\_W} +  T_{q\_G} + T_{comp\_fwd} + T_{comp\_bwd} + \frac{4.5Nd+2.5d^2}{BW}$
- $T'_{mem\_fwd}=\frac{2Nd+2d^2}{BW}$：前向从HBM读取 16-bit 的 X 和 W 的时延
- $T_{q\_X}$：前向过程中对$X$进行 Tile 量化的开销（线速完成）。
- $T_{q\_W}$：对权重$W$量化的开销（与 Baseline 一致）。
- $T_{comp\_fwd}$：前向矩阵乘法计算时延。
- $T'_{mem\_bwd}=\frac{2.5Nd+0.5d^2}{BW}$：反向阶段从 HBM 读取 4-bit $X,W,G_{out}$ 的访存时延。
- $T_{q\_G}$：对输出梯度 $G$ 进行量化的开销。
- $T_{comp\_bwd}$：反向矩阵乘法计算时延。

3. 参数模拟与结论
单层线性计算整体时延缩减 27.95%。
根据在 Blackwell 架构下运行端到端 NVFP4 训练的 profiling 结果：
- 时延分布：通信（访存）与计算的时延比例约为 1:1。
- 量化占比：所有量化算子开销占总时延的 18.4%。
为了进行量化对比，我们设定各参数如下：
- Baseline 方案：总时延 200 单位
  - 计算侧（含量化）：100 单位，其中量化开销 $2(T_{q\_X} + T_{q\_W} +  T_{q\_G})= 200\times 18.4\% = 36.8$ 单位，矩阵计算占 $T_{comp\_fwd} + T_{comp\_bwd} =63.2$ 单位。
  - 访存侧：$\frac{6Nd + 4d^2}{BW}=100$
- 专利方案：总时延 144.1 单位
  - 计算侧：$T_{q\_X} + T_{q\_W} +  T_{q\_G} + T_{comp\_fwd} + T_{comp\_bwd} =81.6$
  - 访存侧：$\frac{4.5Nd+2.5d^2}{BW} \approx 62.5 (N>>d)$

## 六、现有硬件与专利检索记录

本节整理针对本专利 idea 所做的初步检索结果，主要用于后续制作 PPT、撰写专利交底书以及判断避让方向。检索对象包括 NVIDIA GPU、华为 Ascend NPU、Google TPU 的公开量化结构，以及与二维 Tile 量化、训练激活压缩、转置复用、混合精度训练相关的现有专利。

### 1. 本专利 idea 的核心检索对象

本专利当前核心 idea 可概括为：

> 将训练过程中的激活矩阵 $X$ 以二维方形 Tile 为单位进行低精度量化，每个 Tile 共享一个 scale；前向阶段生成的低精度 Tile 以单副本形式持久化存储；反向阶段当 $X$ 以转置形式参与 $\nabla W = X^\top G$ 等计算时，硬件通过 Tile 内转置索引和 scale 对齐读取逻辑复用同一份低精度 Tile 数据和同一份 Tile scale，从而避免保存 16-bit 激活副本或重复执行行/列量化。

围绕该 idea，检索关键词包括：

- NVFP4 rowwise columnwise quantization layout
- Blackwell block scaled GEMM FP4 scale factor
- quantized neural network training and inference transposed activations scale factor
- activation compression backward propagation quantized block floating point
- tile quantization neural network accelerator transpose scale factor
- fine-grained per-vector scaling neural network quantization
- dynamic quantization per tile compute unit
- 华为 Ascend NPU quantization / AscendQuant / dequantization
- Google TPU AQT INT8 quantized training backward propagation

### 2. 当前硬件方案是否支持本专利问题假设

#### 2.1 NVIDIA GPU

NVIDIA Blackwell 的公开文档强烈支持本专利中“维度敏感量化/转置复用困难”的问题判断。

1. **Blackwell 支持 block-scaled FP4/FP8 GEMM**

NVIDIA CUTLASS 文档显示，Blackwell SM100 的 `tcgen05.mma` 支持 `mxf4`、`nvf4` 等 block-scaled MMA。scale 沿 GEMM-K 维度应用，每 16 或 32 个元素共享一个 scale。

参考链接：

- [NVIDIA CUTLASS Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)

这说明当前 NVIDIA FP4/MXFP4/NVFP4 路径主要是面向 GEMM 内积维度的 1D block scaling，而不是天然的二维方形 Tile scaling。

2. **NVFP4 需要 rowwise 和 columnwise 两种量化布局**

NVIDIA Transformer Engine 的 NVFP4 文档明确说明：

- NVFP4 需要 rowwise 和 columnwise quantized tensors；
- rowwise 数据为 `[A, B]`，采用 `1 x 16` horizontal blocks；
- columnwise 数据为 `[B, A]` 的转置布局，并且 scale 也按转置布局存储；
- NVFP4 GEMM 仅支持 TN layout，因此训练中需要专门处理转置方向的数据和 scale。

参考链接：

- [NVIDIA Transformer Engine NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html)

该公开资料直接支持本专利中“前向/反向使用方向不同导致 row/column 量化布局不能自然复用”的技术问题。

3. **scale factor 具有专门硬件友好的 tiled layout**

NVIDIA cuDNN frontend 公开说明，在 Blackwell 上 MXFP8/NVFP4 的 block scaling factors 需要以 `128 x 4` tiled layout 存储，以满足硬件读取效率。该 layout 的目的在于让硬件在处理 tile 时可以连续读取 scale，而不是从分散内存中 gather。

参考链接：

- [NVIDIA cuDNN Frontend: The 128 x 4 Tiled Layout for Block Scaling Factors](https://nvidia.github.io/cudnn-frontend/mxfp8-scale-factor-128x4-layout/)

这说明 NVIDIA 已经非常重视 scale layout 与 Tensor Core 输入读取路径的协同，但公开资料中未见“单副本方形 Tile 激活同时服务原矩阵和转置矩阵”的机制。

4. **需要修正本文档中关于 NVIDIA 的表述**

本文档原有表述“当前硬件量化电路通常集成在乘法器前端”需要更精确。更稳妥的表述应为：

> 当前 GPU 低精度 GEMM 通常消费已经按硬件要求布局好的低精度数据和 scale；量化可能由 Transformer Engine、cuDNN、CUTLASS/Triton kernel、Q/DQ 图优化或 fused kernel 完成，不一定是 Tensor Core 内部乘法器前端自动完成。

#### 2.2 华为 Ascend NPU

华为 Ascend 公开资料可以证明其支持量化/反量化算子和低精度计算，但不足以证明其存在与 NVIDIA NVFP4 相同的 rowwise/columnwise 双布局问题。

1. **Ascend C 提供量化/反量化 API**

Ascend C Operator Development API 中列出如下量化相关接口：

- `AscendQuant`：将 half/float 量化为 int8；
- `AscendDequant`：将 int32 反量化为 half/float；
- `AscendAntiQuant`：执行 fake quantization，例如将 int8 转为 half。

参考链接：

- [Ascend C Quantization and Dequantization APIs](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/ascendcopapi/atlasascendc_api_07_0003.html)

2. **Atlas 量化配置支持权重和数据量化**

华为 Atlas 量化配置文档说明，其模型转换中支持 weight quantification、data quantification、scale、offset 等量化概念。

参考链接：

- [Huawei Atlas Quantization Configuration Overview](https://support.huaweicloud.com/intl/en-us/mcg-atlas500app/atlasmcg_05_c30_0015.html)

3. **结论**

Ascend NPU 可以作为“现有 AI 加速器普遍具有量化/反量化和低精度计算路径”的支撑，但目前公开资料不足以支撑“Ascend 必然需要 rowwise/columnwise 两份量化激活”的具体判断。因此在专利文档和 PPT 中，应将 NVIDIA 作为最强硬件例证，将 Ascend 作为辅助背景，而不是核心证据。

#### 2.3 Google TPU

Google TPU 的公开资料支持“量化训练/推理在硬件上真实存在”，但暂未发现二维 Tile scale 转置复用相关机制。

1. **TPU v5e 支持 INT8 tensor ops**

Google AQT 文档说明，Cloud TPU v5e 可执行 INT8 tensor ops，且 INT8 tensor ops 相比默认 BF16 tensor ops 可获得加速；AQT 支持 forward pass、backward pass 中的 quantized training。

参考链接：

- [Google Cloud Blog: Accurate Quantized Training for TPU v5e](https://cloud.google.com/blog/products/compute/accurate-quantized-training-aqt-for-tpu-v5e)

2. **TPU 公开资料更多强调 MXU / systolic array**

Google TPU 相关专利和文档主要强调 MXU/systolic array、BF16、INT8、FP8 等矩阵计算能力，没有检索到“二维方形 Tile scale，转置后同一 scale 可复用”的公开机制。

3. **结论**

TPU 可作为低精度训练/推理硬件背景，但不是本专利问题的最强例证。

### 3. 高度相关现有专利

#### 3.1 NVIDIA US20230068941A1 / CN115730653A 同族

专利名称：

- [US20230068941A1, Quantized neural network training and inference](https://patents.google.com/patent/US20230068941A1/en)

申请人：NVIDIA Corporation  
优先权日：2021-08-27

该专利是本轮检索中与本文档当前核心 idea 最接近的现有专利。

该专利公开了如下技术点：

- 将 multi-dimensional input tensor 的一部分量化为 quantized matrix；
- 每个 quantized matrix 对应一个 scale factor；
- 不同 scale factors 应用于 tensor 多个维度上的不同 sub-matrices，而不是只应用于单一维度的 vectors；
- 同一个 scale factor 可以用于矩阵中的 sub-matrix 以及其 transposed matrix 中对应的 sub-matrix；
- 这样在训练中使用转置矩阵时，可以减少读取操作并提高效率。

该专利摘要/说明书中明确指出：

> different scale factors are applied to different sub-matrices along multiple dimensions of a tensor, as opposed to different vectors along a single dimension of the tensor. Accordingly, the same scale factor can be applied to both a sub-matrix within a matrix and the corresponding sub-matrix within the corresponding transposed matrix.

这与本文档当前主张的“二维 Tile 共享 scale，原矩阵和转置矩阵复用同一 scale”高度相似。

**风险判断：高。**

如果本文档继续以“二维 Tile 量化 + 转置复用同一 scale”作为主权利要求，可能与 NVIDIA 该专利发生较强重叠。

#### 3.2 US12165038B2, Adjusting activation compression for neural network training

专利名称：

- [US12165038B2, Adjusting activation compression for neural network training](https://patents.google.com/patent/US12165038B2/de)

该专利公开了如下技术点：

- forward propagation 生成的 activation values 可以以 compressed format 暂存到 bulk memory；
- 这些 activation values 可在 backward propagation 中取回使用；
- activation values 可采用 normal precision、quantized format 或 block floating-point format；
- 暂存格式可以比训练中使用的格式进一步压缩；
- compressed format 可以随训练过程动态调整精度。

该专利与本文档中“一次量化后将激活低精度持久化，并在反向传播中取回使用”的部分相关。

**区别：**

该专利重点是 activation compression 的格式调整和训练过程中的压缩策略；本文档重点是二维 Tile 量化、单副本 Tile scale、转置读取复用和硬件索引路径。

**风险判断：中高。**

本文档不应将主创新点写成宽泛的“压缩激活并在反向传播中使用”，而应强调特定的 Tile 描述符、转置读取和 scale 对齐电路。

#### 3.3 US20220067530A1, Fine-grained per-vector scaling for neural network quantization

专利名称：

- [US20220067530A1, Fine-grained per-vector scaling for neural network quantization](https://patents.google.com/patent/US20220067530A1/en)

该专利公开：

- 针对 multi-dimensional tensor，在单一维度内按 vector 划分；
- 每个 vector 具有自己的 scale factor；
- per-vector scaling 可降低量化误差。

该专利是本文档背景中“一维向量级量化”的重要 prior art。

**区别：**

该专利强调 single dimension 的 per-vector scaling；本文档想要保护的是二维方形 Tile 及其转置复用。

**风险判断：中。**

该专利可用于说明本方案与一维 vector scaling 的区别。

#### 3.4 WO2025250327A1, Dynamic quantization

专利名称：

- [WO2025250327A1, Dynamic quantization](https://patents.google.com/patent/WO2025250327A1/en)

该专利公开：

- compute unit/workgroup 可基于 tile 内数据动态确定 scale；
- 可使用历史 scale；
- 对同一 workgroup 中已知的 activation tiles，可在 compute unit/workgroup 内按 per-tile basis 执行动态量化；
- 不需要 second pass。

**区别：**

该专利重点是 per-tile 动态确定 scale，未见“同一低精度 Tile 和 scale 同时服务原矩阵及转置矩阵”的机制。

**风险判断：中。**

本文档不应宽泛保护“per-tile dynamic scale”，而应强调训练前反向复用和硬件转置读取。

### 4. 其他相关资料和专利方向

1. **NVIDIA TensorRT 显式量化**

NVIDIA TensorRT 文档说明，Q/DQ 网络中权重可为 high-precision 或 low-precision quantized type，包括 INT8、FP8、INT4、FP4；预量化权重需要 `IDequantizeLayer` 后接线性算子。

参考链接：

- [NVIDIA TensorRT: Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)

该资料主要支持“低精度权重/激活与 Q/DQ 图优化是现有体系”的背景，不直接影响本文档的二维 Tile 转置复用创新点。

2. **cuDNN Grouped GEMM + Quant**

NVIDIA cuDNN frontend 已有 SM100 上的 grouped GEMM + quant fusion experimental API，支持 block-scaled grouped GEMM、output quantization、per-row gating，主要用于 MoE 工作负载。

参考链接：

- [NVIDIA cuDNN Frontend: Grouped GEMM + Quant Unified](https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.22.0/fe-oss-apis/gemm_fusions/grouped_gemm_quant_unified.html)

该资料说明 NVIDIA 已经在 Blackwell 上做了大量 GEMM + Quant fusion，但不直接公开单副本方形 Tile 激活反向复用机制。

### 5. 当前 patent_1.md 的风险判断

当前文档的核心主张包括：

1. 将量化粒度从一维 row/column vector 改为二维方形 Tile；
2. 每个 Tile 共享一个 scale；
3. 原矩阵和转置矩阵使用同一 Tile scale；
4. 前向阶段将激活 $X$ 量化为低精度 Tile 并持久化；
5. 反向阶段在计算 $X^\top G$ 等操作时复用同一份低精度 Tile。

其中第 1、2、3 点与 NVIDIA US20230068941A1 高度接近；第 4、5 点与 activation compression for backward propagation 相关专利存在一定重叠。

因此，若按当前宽泛写法申请，风险较高。

### 6. 建议避让方向

后续如果继续推进该专利，不建议继续将主权利要求写成：

> 一种二维 Tile 量化方法，其中同一 Tile scale 可用于原矩阵和转置矩阵。

该表述过于接近 NVIDIA 现有专利。

更建议将保护点收窄并转向具体硬件实现：

1. **单副本低精度 Tile 描述符**

   前向阶段生成低精度激活 Tile 后，只保存一份 Tile 数据、一份 Tile scale 和一个 Tile descriptor。descriptor 记录 Tile 的二维尺寸、scale 地址、数据 packing 格式、原始矩阵坐标以及转置读取模式。

2. **Tile 内转置索引发生器**

   在反向计算 $X^\top G$ 时，不重新生成 columnwise quantized tensor，而是由硬件索引发生器对同一物理 Tile 执行 tile-local row/column index remapping。

3. **scale 对齐读取电路**

   对同一 Tile 的正常读取和转置读取，硬件使用同一 Tile scale，但通过 scale alignment logic 将 scale 与转置后的低精度元素流对齐。

4. **面向训练激活的低精度持久化路径**

   将适用对象限定为训练中的 activation stashing：前向中由 BF16/FP16 activation 生成低精度 Tile，写回 HBM；反向中从 HBM 读取同一低精度 Tile，用于权重梯度或输入梯度计算。

5. **与现有 rowwise/columnwise 双布局的区别**

   明确本方案不是保存 rowwise 和 columnwise 两份量化张量，也不是保存 transposed quantized matrix，而是通过同一物理 Tile 的不同读取模式服务两个方向。

### 7. 建议新标题

为了避让 NVIDIA 的“sub-matrix quantization and transposed sub-matrix scale reuse”专利，建议将标题改为更具体的硬件实现方向，例如：

> 一种面向训练激活反向复用的单副本低精度 Tile 描述符与片上转置读取电路

或：

> 一种用于低精度训练的单副本激活 Tile 持久化及转置复用硬件架构

### 8. 建议主权利要求方向

可考虑如下主权利要求框架：

> 一种用于神经网络训练的处理系统，包括低精度 Tile 生成单元、Tile 描述符生成单元、外部存储写回单元、Tile 描述符解析器、Tile 内转置索引发生器和 scale 对齐读取电路。低精度 Tile 生成单元被配置为在前向传播中将高精度激活矩阵的二维 Tile 量化为低精度 Tile；Tile 描述符生成单元被配置为生成描述所述低精度 Tile 的二维尺寸、数据 packing 格式、scale 地址和原始矩阵坐标的描述符；外部存储写回单元被配置为将所述低精度 Tile、对应 scale 和描述符以单副本形式写入外部存储；Tile 描述符解析器被配置为在反向传播中读取所述描述符；Tile 内转置索引发生器被配置为根据反向传播中的矩阵乘法方向，对所述低精度 Tile 内的元素读取顺序执行行列映射；scale 对齐读取电路被配置为将同一 Tile scale 应用于正常读取模式和转置读取模式下的低精度元素流。

### 9. 初步结论

1. 本专利提出的问题是真实的，尤其是 NVIDIA Blackwell NVFP4 训练中 rowwise/columnwise 量化布局并存的问题。

2. 当前 GPU/NPU/TPU 公开资料中，NVIDIA 是最强证据；Ascend 和 TPU 可作为低精度量化硬件背景，但不足以证明相同的 row/column 双布局痛点。

3. 当前 patent_1.md 的宽泛核心 idea 与 NVIDIA US20230068941A1 / CN115730653A 同族专利高度接近，直接申请风险较高。

4. 仍然可能有申请空间，但必须将创新点收窄到具体硬件实现：单副本低精度激活 Tile 持久化、Tile descriptor、Tile 内转置索引发生器、scale 对齐读取电路，以及反向传播中对同一物理 Tile 的复用。

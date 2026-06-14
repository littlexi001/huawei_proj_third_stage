# 专利：一种基于块级量化损失估计的动态精度回退计算架构

## 一、技术问题背景

在 LLM 的低比特（例如 FP4/INT4）推理与训练加速中，主流块量化（block-wise quantization）通常为每个 block 共享一个缩放因子（scale）。该机制的主要失真来源之一是：block 内少量大离群值（outlier）主导 scale，导致其余小幅度元素量化分辨率过粗，出现裁剪为 0、饱和或系统性偏差，从而显著影响模型精度。

在实际执行中，上述失真并非对所有 block 均匀发生。通常只有一部分“异常 block”会出现更强的离群值主导效应：少量大数把 scale 拉大，从而“带走”同一 block 内的大量小数，导致量化后的小数被过度粗化、零化或产生较大偏差。此类异常 block 的存在会在端到端精度上产生放大效应，尤其当异常 block 位于对量化敏感的层/算子/数据语义位置时。

为降低块量化的离群值主导问题，近期出现了多类“分解/平滑”方法（例如 Metis、SmoothQuant 等），通过将原始矩阵变换或分解为两个（或多个）数值范围更小、分布更可控的分量，例如“低秩（或主分量）+ 残差（或被平滑后的分量）”，从而降低低比特量化损失。尽管如此，分解后的各分量在真实推理/训练过程中仍可能出现少量异常 block（例如局部语义激活突变、层/模块敏感性差异、尺度估计抖动等），使得该 block 的量化损失异常增大。

## 二、现有方案

当前硬件量化电路（例如 tensor core/GEMM 路径前端的量化逻辑）通常采用如下执行模式：

1. 从片上 SRAM 或 HBM 读取高精度 block（例如 BF16/FP16）。
2. 在量化单元中为该 block 计算 scale，并生成低比特表示（例如 FP4/INT4）。
3. 直接使用低比特表示进入低精度乘法路径（例如 FP4 GEMM），并输出高精度累加结果（通常为 BF16）。

该方案的局限性在于：当遇到异常 block 时，低比特路径会产生不可接受的量化损失，但系统缺少一种低开销、细粒度的在线判别机制，无法在不显著牺牲吞吐的前提下对异常 block 进行定点修复。

## 三、本发明提出的方案

本发明提出一种“块级量化损失估计 + 动态回退”的软硬协同计算架构。

### 1. 量化前端风险门控机制：量化完成即完成零化统计，并在乘法前选择精度路径

当 SM/计算核心处理某个高精度 block $\mathcal{B}$ 时，量化电路为该 block 计算 scale 并生成低比特 block $\bar{\mathcal{B}}$。与传统方案不同，本发明并不是在所有 block 上无条件进入低精度乘法路径，而是在量化前端同步完成块级风险估计，并在该 block 进入乘法累加路径之前决定其使用低精度路径还是高精度路径。

具体而言，在对 block 内每个元素执行低比特量化的同一流水阶段，硬件同步检测该元素量化后是否为 0，并累计该 block 中被量化到 0 的元素个数。因此，在一个 block 的量化过程结束的同一时刻，硬件已经得到该 block 的零化计数 $c\_{zero}$，并可立即得到块内零化比例 $p\_{zero}(\mathcal{B})$，作为量化损失的 proxy 信号。

其中，设 block size 为 $B$（典型为 32 或 64），对量化后的 block $\bar{\mathcal{B}}={\bar{b}_i}_{i=1}^{B}$，定义：

- $c\_{zero}$：量化后被截断/舍入为 0 的元素个数（统计满足 $\bar{b}\_i=0$ 的个数）。
- 块内零化比例：

$$p\_{zero}(\mathcal{B})=\frac{c\_{zero}}{B}\in\[0,1].$$

随后，硬件在该 block 进入 GEMM micro-tile 乘法累加路径之前执行阈值比较：

$$\text{if } p\_{zero}(\mathcal{B})\le \tau\ \text{then use\_low\_precision}(\bar{\mathcal{B}})\ \text{else use\_high\_precision}(\mathcal{B}).$$

其中 $\tau\in\[0,1]$ 为可编程阈值，可在芯片出厂或系统初始化时写入控制寄存器，也可由 runtime 按层/算子/时延预算进行配置。

在该机制中，同一个 block 只会选择一条乘法路径：若零化比例未超过阈值，则使用生成的低比特 block $\bar{\mathcal{B}}$ 进入 FP4/INT4 乘法路径；若零化比例超过阈值，则直接使用仍保留在寄存器或片上 SRAM 中的原始高精度 block $\mathcal{B}$ 进入 BF16/FP16 高精度路径。由于决策发生在乘法累加之前，因此无需对已经进入 accumulator 的低精度结果进行撤销、覆盖或差分修正。

### 2. 回退策略

当触发回退时，系统对该 block 选择高精度路径，而不是先执行低精度乘法再修正。该 block 保持为 BF16/FP16 表示进入乘法路径，从源头避免低比特表示带来的误差源。

本方案的实施例适用于在线量化场景，例如激活 block 在进入 GEMM 前由 BF16/FP16 实时量化为 FP4/INT4。在该场景中，原始高精度 block 在量化完成和路径选择时仍驻留在寄存器或片上 SRAM 中，因此高精度回退不需要额外从 HBM 重新读取该 block。

### 3. 关键约束：损失估计必须“足够轻量”

本发明的核心要求是：零化计数与阈值比较必须足够轻量，使其不显著增加原有量化前端的关键路径。由于零化计数发生在逐元素量化生成低比特值的同一流水阶段，其主要硬件开销为等零比较、小型计数器或加法树，以及一次阈值比较。

理想情况下，风险门控后的量化前端时延应满足：

$$T\_{\text{quant+risk}} \approx T\_{\text{quant}},$$

其中 $T\_{\text{quant}}$ 为传统块量化前端时延，$T\_{\text{quant+risk}}$ 为加入零化计数和阈值比较后的量化前端时延。这样，系统可以在 block 量化完成的时刻得到精度路径选择信号，并在乘法累加之前完成低精度路径或高精度路径的选择。

## 四、实施例：SM 内块级动态回退的矩阵乘法

以下以线性层矩阵乘 $Y=WX$ 为例，说明本发明在 SM 内的具体运行流程。设 $W\in\mathbb{R}^{m\times k}$，$X\in\mathbb{R}^{k\times n}$。为便于说明，假设量化发生在 $X$ 的内积维度 $k$ 上，即将 $X$ 沿 $k$ 维按 block size $B$（32 或 64）切分为多个 block，并在每个 block 上独立计算 scale 与零化比例。

### 1. 阈值的预设与配置

1. 系统为该算子或该层配置一个零化比例阈值 $\tau\in\[0,1]$。
2. $\tau$ 写入一个可由硬件读取的控制寄存器或执行描述符字段（例如 per-layer/per-operator 的配置项）。
3. 为避免软件介入，$\tau$ 可采用一个全局默认值；或在部署阶段离线扫参得到推荐值后固化到配置中。

### 2. 单个 block 的量化、统计与路径选择

当计算核心即将处理 $X$ 的某个高精度 block $\mathcal{B}={b\_i}\_{i=1}^{B}$（BF16/FP16）时，量化单元执行如下操作：

1. **计算 scale**：基于 block 内统计（例如最大绝对值）计算缩放因子 $s\_B$。
2. **逐元素量化并计数零化**：对每个元素 $b\_i$ 生成对应的低比特值 $\bar{b}\_i$。在生成 $\bar{b}\_i$ 的同一流水级，硬件并行进行等于 0 检测：
   - 若 $\bar{b}_i=0$，则将计数器 $c_{zero}$ 加 1。
   - 否则计数器不变。
     该计数过程可用加法树或小型计数器阵列实现，不要求额外的乘法或开方运算。
3. **得到零化比例**：在 block 量化完成后，计算

$$p\_{zero}(\mathcal{B})=\frac{c\_{zero}}{B}.$$

由于 $B$ 为固定常数（32/64），该除法可用移位或乘常数实现，延迟可控。
4. **阈值比较与决策**：在 block 量化完成的同一时刻，硬件已经得到 $p\_{zero}(\mathcal{B})$，并立即与阈值 $\tau$ 比较，形成 1-bit 精度路径选择信号 $d(\mathcal{B})$：

$$d(\mathcal{B})=\mathbb{I}\[p\_{zero}(\mathcal{B})\le\tau].$$

5. **乘法前路径选择**：该决策信号在 block 进入 GEMM micro-tile 乘法累加路径之前生效：
   - 若 $d(\mathcal{B})=1$，则将低比特 block $\bar{\mathcal{B}}$ 及其 scale $s\_B$ 送入 FP4/INT4 低精度乘法路径；
   - 若 $d(\mathcal{B})=0$，则旁路低精度乘法路径，直接将仍驻留在寄存器或片上 SRAM 中的原始高精度 block $\mathcal{B}$ 送入 BF16/FP16 高精度乘法路径。

### 3. 低精度路径与高精度回退路径的互斥执行

1. **若 $d(\mathcal{B})=1$（零化比例不超阈值）**：
   - 认为该 block 的量化失真可接受；
   - 硬件使用低精度 block $\bar{\mathcal{B}}$ 计算该 block 对输出 accumulator 的贡献。
2. **若 $d(\mathcal{B})=0$（零化比例超阈值）**：
   - 认为该 block 为异常 block；
   - 硬件不将该 block 的低比特表示送入低精度乘法路径，而是直接对原始高精度 block $\mathcal{B}$ 启动 BF16/FP16 高精度路径；
   - 高精度路径产生该 block 对输出 accumulator 的贡献。

回退时，该 block 的输入保留为 BF16/FP16 表示进入乘法路径，从源头避免低比特引入的误差。

在硬件实现上，低精度路径与高精度路径对同一个 block 互斥执行。由于路径选择发生在乘法累加之前，系统不需要暂存低精度推测结果，也不需要在输出融合侧执行 $Y\_{hi}-Y\_{low}$ 的差分修正。只要零化计数和阈值比较不显著增加量化前端时延，回退仅在少量异常 block 上发生时，整体吞吐损失可被控制在较小范围内。

## 五、专利效果建模（示例）

本节用于量化本发明在“精度收益—系统开销”上的效果。由于回退判据 $p\_{zero}(\mathcal{B})$ 与阈值 $\tau$ 已在本发明方案中给出，本节重点描述如何通过实验或仿真得到回退率、时延估计与输出误差之间的关系。

### 5.1 建模变量与关键假设

- 设 block 总数为 $M$，在给定阈值 $\tau$ 下的回退比例为 $p(\tau)$。
- 设低比特（例如 FP4）路径的单位代价为 1，高精度 BF16 回退路径的相对代价为 $\alpha$（例如假设 BF16 为 FP4 的 4 倍，则 $\alpha=4$）。
- 假设零化比例统计与阈值比较和块量化同流水完成，即 $T\_{\text{quant+risk}}\approx T\_{\text{quant}}$，不会显著增加量化前端关键路径。

### 5.2 以线性层 $Y=WX$ 为例的仿真实验设计

本实验用于生成随阈值 $\tau$ 变化的三条曲线：回退率、时延估计、输出量化损失。

**实验对象**

- 选择一个具体矩阵乘：$W\in\mathbb{R}^{m\times k}$、$X\in\mathbb{R}^{k\times n}$。
- 计算 BF16（或 FP32）参考输出：$Y\_{ref}=WX$。

**实验实现（推荐从量化 $X$ 开始）**

1. 选定 block size $B$（32 或 64），以及低比特格式（例如 FP4）。
2. 按 GEMM 的内积维度对 $X$ 做 block-wise 量化：得到每个 block 的 $\bar{\mathcal{B}}$ 与其 scale。
3. 对每个 block 统计 $p\_{zero}(\mathcal{B})$，并按阈值 $\tau$ 决定该 block 的路径：
   - 若 $p\_{zero}(\mathcal{B})\le\tau$：该 block 使用低比特反量化值 $\tilde{\mathcal{B}}$（由 $\bar{\mathcal{B}}$ 与 scale 反量化得到的 BF16 值）。
   - 若 $p\_{zero}(\mathcal{B})>\tau$：该 block 回退，使用原始 BF16 block $\mathcal{B}$。
4. 将所有 block 拼回得到 $X\_{mix}(\tau)$，计算：

$$Y\_{mix}(\tau)=W X\_{mix}(\tau).$$

5. 扫描一组阈值 $\tau\in\[0,1]$（例如 ${0, 1/64, 2/64, \ldots, 1}$ 或更稀疏的网格），重复步骤 3–4。

### 5.3 实验需输出的指标（随 $\tau$ 的曲线）

**1) 回退率（Fallback Rate）**

设总 block 数为 $M$，回退 block 数为 $M\_{fb}(\tau)$，则：

$$p(\tau)=\frac{M\_{fb}(\tau)}{M}.$$

建议同时输出 $p\_{zero}$ 的分布（直方图或分位数），以证明单一阈值在不同层/算子上的可迁移性。

**2) 时延估计（Latency）**

在零化计数与阈值比较不显著增加量化前端关键路径，且每个 block 在进入乘法前只选择低精度或高精度其中一条路径的假设下，时延主要由少量高精度回退 block 引入。若高精度 BF16 计算相对低比特 FP4 的慢速因子为 $\alpha$（例如 $\alpha=4$），则端到端时延比可用下式近似：

$$\mathrm{LatencyRatio}(\tau)\approx 1+p(\tau)(\alpha-1).$$

代入 $\alpha=4$ 得：

$$\mathrm{LatencyRatio}(\tau)\approx 1+3p(\tau).$$

**3) 乘法结果的量化损失（Output Quantization Loss）**

建议使用相对 Frobenius 误差作为主指标：

$$\mathrm{Err}(\tau)=\frac{|Y\_{mix}(\tau)-Y\_{ref}|_F}{|Y_{ref}|\_F}.$$

同时建议输出 tail-case 指标（用于体现“异常 block 定点修复”的价值）：

- 将 $Y$ 按 token/列切分，统计每列相对误差的 P95/P99。

### 5.4 预期结论形式

通过扫描 $\tau$，得到三条曲线 $p(\tau)$、$\mathrm{LatencyRatio}(\tau)$、$\mathrm{Err}(\tau)$，并可绘制 Pareto 前沿：

- 在较小回退率下（小 $p$），若异常 block 的误差贡献占比较高，$\mathrm{Err}(\tau)$ 将快速下降。
- 若 $p\_{zero}$ 分布在绝大多数 block 上较低，则可以选择一个统一阈值 $\tau$ 在不同层/算子上取得相近的回退率与误差收益。

## 六、现有专利检索与避让建议

本节记录针对本专利 idea 进行的初步现有专利检索结果。检索范围主要包括 Google Patents 中的 US、WO、CN 公开文本，并以中国专利文本和国际专利族作为主要参考。CNIPA 官网检索系统更适合人工交互式检索，公开网页对搜索引擎抓取不稳定，因此本节结果仅作为后续撰写和避让的技术参考，不构成法律意见。

### 6.1 本专利 idea 的检索对象

本专利的核心技术点可概括为：

> 在 block-wise FP4/INT4 在线量化过程中，同步统计该 block 中量化后为 0 的元素数量，量化结束时得到零化比例 $p\_{zero}$，并在该 block 进入 GEMM micro-tile 乘法累加路径之前，根据阈值决定该 block 走低精度路径还是高精度 BF16/FP16 路径。低精度路径和高精度路径互斥执行，不需要先执行低精度再修正。

围绕该核心点，检索关键词包括：

- block quantization + fallback / high precision
- quantization error + threshold + neural network accelerator
- zero count / zero ratio + quantization
- mixed precision + block / group + dynamic bit-width
- residual quantization + quantization error threshold
- compensation instruction + quantization error
- 块量化、动态精度、混合精度、量化误差、神经网络加速器、零值计数

### 6.2 高相关现有专利

| 专利 | 相关点 | 与本方案的区别 | 风险等级 |
|---|---|---|---|
| [US20200193273A1 / US11586883B2, Microsoft, Residual quantization for neural networks](https://patents.google.com/patent/US20200193273A1/en) | 该专利将 normal-precision tensor 转成 quantized tensor，同时产生 residual tensor；使用 conversion error、layer number、exponent value、layer type 等 input metrics 判断是否使用 residual tensor，从而获得更高精度。文中还包含 control hardware，用于选择 higher/lower precision。 | 该方案是“量化张量 + residual tensor”的选择与组合，偏 residual correction；本方案是“量化过程中统计零化比例，并在乘法前互斥选择 FP4/INT4 或 BF16/FP16 路径”，不生成 residual tensor，也不将 residual 结果与 quantized 结果相加。 | 高 |
| [CN113902108B，一种量化位宽动态选择的神经网络加速硬件架构及方法](https://patents.google.com/patent/CN113902108B/zh) | 该专利以块/组为单位划分神经元，使用动态量化预测控制器，通过高精度部分计算结果判断是否执行低精度计算，并根据阈值决定块内组的执行数量和位置。 | 该方案先执行高精度部分，再判断是否执行低精度部分；判断依据是组最大值、L1 范数、可训练阈值和稀疏度。本方案是在在线量化时统计低比特零化数量，在 GEMM 乘法前选择 FP4/INT4 或 BF16/FP16，且不是“高精度部分 + 低精度部分相加”。 | 高 |
| [CN112119407A, Low precision deep neural network enabled by compensation instructions](https://patents.google.com/patent/CN112119407A/en) | 该专利涉及 quantization error compensation instruction，其中包括 zero compensation bit，用于指示估计量化误差是否小于阈值；硬件根据补偿指令修正乘法/点积。 | 该方案给量化值附带补偿指令，并通过 compensation circuit 修正低精度结果。本方案不是误差补偿指令，也不修正低精度乘积，而是在乘法前选择高/低精度路径。 | 中高 |

### 6.3 中等相关现有专利

| 专利 | 相关点 | 与本方案的区别 | 风险等级 |
|---|---|---|---|
| [CN114626516A，一种基于对数块浮点量化的神经网络加速系统](https://patents.google.com/patent/CN114626516A/zh) | 涉及块浮点量化、在线确定当前激活值量化分块的共享指数、硬件转换单元和计算单元。 | 该方案关注 log block floating-point 量化和硬件加速系统，未见基于零化比例的动态高精度回退。 | 中 |
| [CN112906883B，用于深度神经网络的混合精度量化策略确定方法和系统](https://patents.google.com/patent/CN112906883B/zh) | 通过推理精度阈值和推理时间阈值，为不同层筛选混合精度策略。 | 该方案是离线或策略层面的 layer-wise 混合精度搜索，不是 SM 内 block 级在线路径选择。 | 中低 |
| [US20240062059A1, Neural network layer optimization](https://patents.google.com/patent/US20240062059A1/en) | 按 layer 的 accuracy improvement 和 latency degradation 选择不同 bit precision。 | 该方案是 layer-level 配置优化，不涉及量化流水线内零化计数、block 级门控和 GEMM 前选路。 | 中低 |
| [US20200401895A1, Neural network hardware accelerator system with zero-skipping](https://patents.google.com/patent/US20200401895A1/en) | 涉及 zero-skipping、零权重和结构化剪枝硬件。 | 该方案处理的是稀疏/剪枝后的零跳过，不是统计“量化导致的零化比例”来决定精度路径。 | 低 |

### 6.4 需要避开的宽泛表述

结合上述检索结果，后续撰写权利要求时应避免将主权利要求写得过宽，以免落入已有混合精度、误差补偿或 residual quantization 的保护范围。

应避免的表述包括：

- “根据量化误差阈值选择高精度或低精度计算”。
  - 风险：容易与 Microsoft residual quantization 相关专利重叠。

- “对神经网络中的块动态选择量化位宽”。
  - 风险：容易与 CN113902108B 以及其他 block/group 级混合精度量化专利重叠。

- “量化误差小于阈值则不补偿，大于阈值则补偿”。
  - 风险：容易与 CN112119407A 的 compensation instruction 思路重叠。

- “根据层精度收益和时延损失选择精度”。
  - 风险：已有多件 layer-wise mixed precision 相关专利。

### 6.5 建议强化的差异化保护点

为提高本专利 idea 与现有方案的区分度，建议将主创新点收束到以下更具体、更硬件化的特征上：

1. **零化计数在量化同一流水阶段完成**

   强调本方案不是离线误差评估，不是 layer-wise 搜索，也不是 residual error metric，而是在生成每个低比特元素的同一流水级执行等零检测，并在 block 量化完成时得到 $c\_{zero}$。

2. **决策发生在 GEMM micro-tile 乘法累加之前**

   明确该 block 在进入 accumulator 之前即被路由到 FP4/INT4 路径或 BF16/FP16 路径。两条路径对同一 block 互斥执行，不执行低精度结果修正或 residual 累加。

3. **高精度源数据来自在线量化时仍驻留的寄存器或片上 SRAM**

   优选实施例应聚焦在线激活量化：原始 BF16/FP16 block 在量化完成和路径选择时仍保留在寄存器或片上 SRAM 中，因此无需从 HBM 重新读取高精度副本。这一点区别于“预存 quantized tensor + residual tensor”的方案。

4. **判据为量化后零化比例，而不是一般 quantization error**

   主权利要求中应优先写成：基于量化后低比特元素等于零的数量或比例生成路径选择信号。从属权利要求再扩展到 saturation rate、clip rate、scale range、局部重构误差估计等其他 proxy 信号。

5. **硬件结构应写成量化前端风险门控器**

   可使用如下模块表述：

   - zero-count accumulator；
   - block risk gate；
   - precision path selector；
   - quantization-front-end routing circuit；
   - per-layer threshold register / execution descriptor field。

   这样本方案更像 SM/GEMM 前端硬件结构，而不是普通软件量化算法。

### 6.6 建议主权利要求方向

后续正式撰写时，主权利要求可以考虑围绕如下技术组合展开：

> 一种用于神经网络矩阵乘法的处理单元，包括块量化单元、零化计数单元、阈值比较单元和精度路径选择单元。块量化单元被配置为将寄存器或片上存储中的高精度输入 block 量化为低比特 block；零化计数单元被配置为在生成低比特元素的同一流水阶段统计低比特元素中等于零的元素数量；阈值比较单元被配置为在所述 block 量化完成时基于所述数量生成路径选择信号；精度路径选择单元被配置为在该 block 进入矩阵乘法累加路径之前，根据所述路径选择信号将低比特 block 送入低精度乘法路径，或将仍驻留于寄存器或片上存储中的原始高精度 block 送入高精度乘法路径。

### 6.7 初步判断

本专利 idea 所处区域已有较多“混合精度、误差阈值、硬件控制、高低精度选择”的现有专利，尤其需要认真避让 Microsoft 的 residual quantization 专利族和 CN113902108B 的动态块/组混合精度硬件方案。

但在本轮检索中，尚未发现完全相同的组合：

> 在线 block 量化过程中同步统计量化后零值数量，量化结束即形成零化比例，并在 GEMM micro-tile 进入 accumulator 前互斥选择低精度或高精度路径。

因此，该方向仍有可写空间。后续撰写时建议将主创新点进一步收束到“零化计数同流水 + 乘法前互斥路由 + 在线激活 block 高精度源保留”这三件事上。

### 6.8 参考文献与检索资料清单

下表整理本轮检索中被记录和引用的主要专利文献。字段中的“国籍/地区”按专利公开文本所属国家/地区或公开机构统计；“相关位置”用于后续撰写 PPT、交底书和答复审查意见时快速定位该文献与本方案的关系。

| 国籍/地区 | 专利号 / 文献号 / 标题 | 公开日期 | 文档来源 | 相关页码、图表编号或章节号 |
|---|---|---:|---|---|
| 美国 | `US20200193273A1` / `US11586883B2`，`Residual quantization for neural networks` | A1 公开：2020-06-18；B2 授权：2023-02-21 | [Google Patents](https://patents.google.com/patent/US20200193273A1/en) | 摘要；`FIG. 1` 计算系统；`FIG. 8` 量化与 residual 生成；`FIG. 9-10` 量化张量与 residual 张量；`FIG. 11-13` 量化/残差选择流程；Claims 中关于 conversion error、residual tensor 和 higher/lower precision control hardware 的限定 |
| 中国 | `CN113902108A` / `CN113902108B`，`一种量化位宽动态选择的神经网络加速硬件架构及方法` | A 公开：2022-01-07；B 授权：以 CNIPA / Google Patents 页面为准 | [Google Patents](https://patents.google.com/patent/CN113902108B/zh) / CNIPA | 摘要；说明书中“动态量化预测控制器”“高精度部分计算结果”“低精度计算执行数量和位置”等段落；附图中硬件架构和方法流程图；权利要求中关于块/组划分、阈值判断和动态位宽选择的限定 |
| 中国 | `CN112119407A`，`Low precision deep neural network enabled by compensation instructions` | A 公开：2020-12-22 | [Google Patents](https://patents.google.com/patent/CN112119407A/en) | 摘要；说明书中 quantization error compensation instruction、zero compensation bit、compensation circuit 相关段落；附图中低精度 DNN 计算、补偿指令和硬件补偿流程；权利要求中关于补偿指令修正乘法/点积结果的限定 |
| 中国 | `CN114626516A`，`一种基于对数块浮点量化的神经网络加速系统` | A 公开：2022-06-14 | [Google Patents](https://patents.google.com/patent/CN114626516A/zh) / CNIPA | 摘要；说明书中 log block floating-point、共享指数、在线量化、硬件转换单元和计算单元相关段落；附图中的系统结构图和量化/计算流程图；用于说明“块浮点量化硬件”是相关背景，但未见零化比例 fallback |
| 中国 | `CN112906883A` / `CN112906883B`，`用于深度神经网络的混合精度量化策略确定方法和系统` | A 公开：2021-06-04；B 授权：以 CNIPA / Google Patents 页面为准 | [Google Patents](https://patents.google.com/patent/CN112906883B/zh) / CNIPA | 摘要；说明书中混合精度量化策略、推理精度阈值、推理时间阈值、layer-wise 搜索和策略筛选相关段落；权利要求中关于不同层精度策略确定的限定 |
| 美国 | `US20240062059A1`，`Neural network layer optimization` | A1 公开：2024-02-22 | [Google Patents](https://patents.google.com/patent/US20240062059A1/en) | 摘要；`FIG. 1` 神经网络层优化系统；`FIG. 2-3` layer precision / accuracy-latency tradeoff 流程；说明书中 accuracy improvement、latency degradation、bit precision selection 相关段落；用于说明 layer-level 精度配置优化是已有方向 |
| 美国 | `US20200401895A1`，`Neural network hardware accelerator system with zero-skipping` | A1 公开：2020-12-24 | [Google Patents](https://patents.google.com/patent/US20200401895A1/en) | 摘要；`FIG. 1` accelerator system；`FIG. 2-4` zero-skipping / sparse weight handling 相关结构；说明书中 zero weight、zero-skipping、pruning 和稀疏计算相关段落；用于区分“跳过已有零值/剪枝零值”与本方案“统计量化导致的零化比例并选择精度路径” |

本轮检索中，人工筛选并记录的主要专利文献为 **7 组**。其中，高相关文献 **2 组**（Microsoft residual quantization、CN113902108B 动态量化位宽选择），中高相关文献 **1 组**（CN112119407A 补偿指令），中等或背景相关文献 **4 组**（块浮点量化、layer-wise 混合精度策略、layer optimization、zero-skipping）。上述数量为人工筛选后的有效相关文献数量，不等同于 Google Patents 或 CNIPA 中各检索式的原始命中总数。

## 七、专利价值说明：专利价值类别及商业价值

### 7.1 专利价值类别

1. **系统类 / 装置类价值**

   本发明可保护一种面向低比特矩阵乘法的硬件处理装置，包括块量化单元、零化计数单元、阈值比较单元、精度路径选择单元、低精度乘法路径和高精度回退路径。该装置可集成于 AI 训练与推理芯片的 SM、Tensor Core、Cube 单元、矩阵乘法阵列或 GEMM 前端数据准备模块中，用于在 block 进入矩阵乘法累加路径之前完成风险判断和精度路由。

2. **方法类价值**

   本发明可保护一种神经网络矩阵乘法中的动态精度选择方法，即在在线 block-wise 量化过程中同步统计量化后为零的元素数量，基于零化比例与阈值比较结果，在乘法累加前互斥选择低精度 FP4/INT4 路径或高精度 BF16/FP16 路径。该方法覆盖量化、零化统计、风险判别、路径选择和矩阵乘法执行的完整流程。

3. **架构平台类价值**

   本发明可作为低比特训练与推理芯片中的精度风险门控架构，用于在不显著增加量化前端关键路径的前提下，对少量异常 block 进行定点高精度回退。该架构不局限于单个线性层，可扩展到 Transformer MLP、Attention projection、MoE expert GEMM、训练反向传播 GEMM 以及其他低比特矩阵乘密集场景。

### 7.2 商业价值

1. **增强我司 AI 训练与推理芯片的产品竞争力**

   低比特 FP4/INT4 计算是提升 AI 芯片算力密度和能效的重要方向，但异常 block 导致的量化误差会制约低比特训练和高精度推理落地。本发明通过零化比例风险门控和少量高精度回退，在保持大部分 block 走低精度路径的同时修复精度尾部风险，有助于提升芯片在大模型训练、微调和推理场景中的有效精度、吞吐和能效表现。

2. **形成面向友商演进趋势的前瞻布局**

   当前主流 GPU/NPU/TPU 厂商公开资料多集中在 block-scaled GEMM、低精度 Tensor Core/NPU 计算单元、量化融合 kernel、编译器混合精度策略和 scale layout 优化等方向。公开资料中尚未充分披露“在线量化同流水零化计数 + GEMM 累加前互斥高/低精度选路”的完整硬件机制。本发明可围绕该差异化方向形成专利壁垒，为后续芯片架构演进和竞争防御提供支撑。

3. **支撑数据中心 AI 加速的精度—成本平衡**

   数据中心大模型训练和推理需要在吞吐、能耗、HBM 带宽和模型精度之间取得平衡。本发明使系统能够只对少量高风险 block 启动 BF16/FP16 回退，而不是整体提高精度或全量保存高精度中间结果，从而在可控时延开销下提升低比特计算稳定性。该能力有助于降低大模型部署成本，提升数据中心 AI 加速卡在高负载场景下的性能/功耗竞争力。

## 八、可发现性说明

本节从专利运营和侵权取证角度说明：若竞争对手在 AI 训练或推理芯片中实现类似“零化比例风险门控 + 动态精度回退”机制，可能通过哪些公开或黑盒方式观察到相关痕迹。

### 8.1 公开资料与专利文本

若竞争对手采用类似方案，可能会在产品白皮书、低精度训练指南、芯片架构文档、开发者手册、GEMM kernel 说明、编译器优化文档或专利申请中披露相关线索。重点关注其是否提到如下表述：

- block-level / group-level quantization risk estimation；
- zero count、zero ratio、zero-rate、sparsification after quantization；
- quantization loss proxy、block risk gate、precision fallback；
- 在量化过程中统计低比特零值数量；
- 根据 block 风险信号选择 FP4/INT4 或 BF16/FP16 计算路径；
- 低精度路径与高精度路径在矩阵乘法前互斥执行。

尤其需要关注对“量化误差估计”或“混合精度选择”的具体硬件实现描述。如果其判据并非完整重构误差，而是量化后零值数量、饱和数量、裁剪数量或类似轻量 proxy，则与本方案的接近程度较高。

### 8.2 硬件接口、编译器与运行时痕迹

若该方案被产品化，通常需要在硬件控制字段、算子描述符、编译器 IR、runtime kernel 参数或低精度 GEMM API 中体现相应控制信息。例如：

- per-layer / per-operator 的零化比例阈值；
- block size、量化格式、fallback enable 标志；
- 高精度回退路径开关或 mixed precision fallback 配置；
- 量化前端统计信息字段，例如 zero count、clip count、saturation count；
- kernel trace 或 runtime 日志中显示部分 block 走 BF16/FP16 fallback；
- 编译器在低精度 GEMM 前插入或融合风险门控逻辑。

即使硬件内部细节不公开，也可能通过算子库接口、调试环境变量、编译器生成的 kernel 名称、性能计数器或 profiling event 观察到“block 级风险统计”和“高精度回退”相关痕迹。

### 8.3 Profiling 与黑盒行为特征

可以通过 profiling 和黑盒实验间接判断是否存在类似机制。典型测试方法包括构造具有不同 outlier 分布和零化比例的输入 block，观察芯片在低精度 GEMM 中的精度、时延和硬件计数器变化。

若某芯片存在类似机制，可能表现出以下特征：

- 当输入 block 的量化后零化比例超过某一阈值时，输出误差突然下降或保持稳定；
- 高风险 block 增多时，时延或能耗呈分段式上升，而不是连续平滑变化；
- 同一低精度算子在不同 outlier 分布下出现可重复的 fallback 开销；
- profiling 中低精度 tensor op 占比下降，同时 BF16/FP16 计算或相关混合精度路径占比上升；
- 在不显著增加 HBM 读写的情况下，异常 block 的输出误差得到修复，说明高精度源数据可能来自寄存器或片上 SRAM 中仍保留的原始 block。

这些黑盒特征无法单独证明侵权，但可作为进一步分析硬件接口、编译器行为和专利文本的线索。

### 8.4 最有价值的证据类型

最有价值的证据包括：

1. 公开文档或专利中明确描述“在 block 量化过程中统计量化后零值数量或比例，并基于该统计结果选择高/低精度矩阵乘法路径”。
2. 编译器、runtime 或 kernel API 中出现 zero-count threshold、fallback threshold、block risk gate、precision selector 等字段。
3. profiling 证明部分 block 在低精度 GEMM 前被路由到 BF16/FP16 路径，且低精度和高精度路径对同一 block 互斥执行。
4. 黑盒实验显示输出误差与零化比例阈值存在明显相关性，并伴随可解释的 fallback 时延开销。
5. 调试工具或性能计数器显示量化前端存在 zero count / clip count / saturation count 等统计事件，并用于控制精度路径选择。

## 九、可规避性与上位保护建议

### 9.1 潜在可规避方式

1. **硬件命名和模块划分规避**

   竞争对手可能不使用“零化计数单元”“风险门控器”或“精度路径选择器”等名称，而是将相关功能描述为量化前端、异常检测单元、混合精度控制器、数据路由器、GEMM scheduler、微码控制逻辑或编译器生成的硬件控制流程。如果权利要求过度限定模块名称，容易被通过模块重命名或功能合并规避。

2. **判据信号规避**

   竞争对手可能不直接使用“零化比例”作为唯一判据，而改用量化后非零数量、裁剪比例、饱和比例、scale 范围、局部动态范围、局部重构误差估计、指数分布特征或多个 proxy 信号组合。如果主权项只限定“零化比例”，保护范围可能偏窄。

3. **执行动作规避**

   竞争对手可能不表述为“回退到 BF16/FP16 路径”，而是描述为提升 bit-width、选择 higher precision compute mode、选择 alternate GEMM kernel、旁路低精度 Tensor Core、使用 mixed precision micro-tile 或重新路由到高精度计算单元。如果权利要求只写固定精度格式或固定路径名称，容易被通过术语变化规避。

4. **输入数据和应用范围规避**

   竞争对手可能不限定为激活 block 或线性层 $Y=WX$，而是将机制用于权重 block、梯度 block、Attention 中间张量、MoE expert 输入、KV cache 或任意矩阵乘法操作数。如果权利要求只覆盖某一类张量或某个算子，可能被通过应用对象变化规避。

5. **物理位置和时序规避**

   竞争对手可能将风险统计放在量化单元旁路、GEMM scheduler、片上 SRAM 读取阶段或编译器生成的 fused kernel 中，而不强调“量化同一流水阶段”。也可能在 micro-tile 调度前、warp/block 调度前或 accumulator 前的其他时刻完成选路。如果权利要求将物理位置或时序写得过窄，可能被等效实现绕开。

### 9.2 上位保护建议

1. **硬件机制上位化**

   主权项中建议使用“量化处理单元、统计单元、阈值判断单元、精度路径控制单元、矩阵乘法计算单元”等较上位术语，而不局限于 zero-count accumulator 或 block risk gate 的具体名称。这样既覆盖专用硬件电路，也覆盖由微码、硬件描述符、编译器调度或 runtime 配合实现的等效结构。

2. **反馈信息 / 风险信号上位化**

   主权项可优先保护“基于量化后低比特元素分布特征生成的块级风险信息”，并在从属权利要求中限定该风险信息包括量化后为零的元素数量或比例。这样既能突出本方案的优选零化比例判据，又能覆盖通过非零数量、饱和数量、裁剪数量、scale 范围或局部误差 proxy 实现的等效风险门控。

3. **执行动作上位化**

   不宜仅写“选择 FP4/INT4 或 BF16/FP16”，建议上位表述为“根据块级风险信息选择第一精度计算路径或第二精度计算路径，其中第二精度计算路径的精度高于第一精度计算路径”。从属权利要求中再限定第一精度为 FP4/INT4，第二精度为 BF16/FP16。这样可覆盖 FP8、INT8、MXFP4、NVFP4 或其他未来低比特格式。

4. **物理范围与适用对象上位化**

   主权项中可将处理对象上位为“神经网络矩阵乘法中的输入数据块或操作数数据块”，而不是仅限定为激活矩阵 $X$。计算场景可上位为“神经网络中的矩阵乘法或张量乘法操作”，从属权利要求再限定线性层、Transformer MLP、Attention projection、训练反向传播、MoE expert GEMM 等优选实施例。

5. **时序关系上位化但保留关键边界**

   本方案的关键边界是“在该数据块进入矩阵乘法累加路径之前完成路径选择”，而不是必须限定在某个具体硬件流水级。建议主权项强调“在乘法累加前生成路径选择信号并执行互斥路由”，从属权利要求再限定“在生成低比特元素的同一流水阶段执行零化计数”。这样既保留核心区别点，又减少被物理流水实现差异规避的风险。

6. **核心组合保护**

   主权项应重点保护如下技术组合：

   > 对神经网络矩阵乘法的数据块执行低比特量化；基于量化后低比特元素的分布特征生成块级风险信息；在该数据块进入矩阵乘法累加路径之前，根据块级风险信息将该数据块互斥路由至较低精度计算路径或较高精度计算路径。

   在此基础上，从属权利要求再收束到“量化后零化数量/比例”“量化同流水统计”“原始高精度 block 仍驻留于寄存器或片上 SRAM”“无需 residual tensor 或低精度结果补偿”等更具体的差异化特征。


发明背景
现有技术路径（GPU/TPU/NPU）
本发明技术方案（概述1页、模块与工作流程一页，含图；实施例2～3页，含图）
本发明的技术保护点
技术效果
专利价值说明：专利价值类别及商业价值
可发现性：公开资料与专利文本；硬件接口、编译器与运行时痕迹；prifiling 与黑盒特征；最有价值的证据类型
本发明的可规避性和上位保护
公开文件检索公式，即专利检索关键词组合
各公开文档的国别、专利号、日期、来源、相关的段落或页数或图片编号（做成一页表格）
与三个重点现有专利的对比，包含该专利的方案及其与本专利的不同之处，关键是体现出为什么我们这个专利还是值得申请；

# Adaptive-Metis 稀疏标定与约束搜索方案

## 1. 阶段目标与问题定义

本阶段工作的首要目标是满足 SOW 对低精度模型性能的约束：

- HIF8：任务性能损失小于 `0.5%`；
- HIF4：任务性能损失小于 `1%`。

在满足上述性能要求的前提下，进一步寻找计算成本尽可能低的 Metis 配置。对于当前全局 rank 配置，权重侧 rank 记为 `kp`，激活侧 rank 记为 `ka`：

- `kp` 决定权重矩阵中以高精度保留的主谱方向数量；
- `ka` 决定激活矩阵中以高精度保留的主谱方向数量；
- 未被高精度 head 覆盖的 residual 谱空间进入 HIF4 或 HIF8 低精度计算。

因此，本阶段的配置选择问题形式化为：

```math
(k_a^*,k_p^*)
=
\arg\min_{k_a,k_p}
T(k_a,k_p)
```

满足：

```math
\Delta P(k_a,k_p)
\le
\epsilon_f
```

其中：

- `T(ka,kp)` 表示配置对应的 latency 或硬件成本；
- `ΔP(ka,kp)` 表示相对高精度 baseline 的任务性能损失；
- `ε_f` 表示数据格式 `f` 对应的 SOW 性能上限，HIF8 为 `0.5%`，HIF4 为 `1%`。

若当前阶段尚未获得目标硬件上的完整 latency profile，可先以：

```math
C(k_a,k_p)
=
c_a k_a+c_p k_p
```

作为 rank cost。其中 `c_a` 和 `c_p` 后续由目标硬件实测结果校准。当二者暂取相同权重时，成本简化为 `ka+kp`。

本阶段不再以穷举全部配置的真实任务性能作为部署搜索方式。完整 sweep 仅用于方法开发阶段构造 oracle，对实际交付流程而言，目标是用少量 calibration task evaluation 找到满足性能要求的最低成本配置。

## 2. Performance 与 Latency 建模

### 2.1 基本变化关系

Metis 在谱空间中保留 rank-`k` 的高精度 head，并对剩余 residual 进行低精度量化。随着 `ka/kp` 增大，更多主谱方向被高精度保留，residual 中承载的重要信息通常减少。因此整体上存在以下趋势：

```math
k_a,k_p \uparrow
\quad\Longrightarrow\quad
\Delta P(k_a,k_p) \downarrow
```

与此同时，更高的 rank 会增加低秩 head 的分解、存储与计算成本：

```math
k_a,k_p \uparrow
\quad\Longrightarrow\quad
T(k_a,k_p) \uparrow
```

上述关系在真实任务 accuracy 上不要求逐点严格单调。有限样本统计波动、不同谱方向的任务敏感性以及 HIF block scale 的变化，都可能导致局部非单调现象。但从整体响应面看，较高 rank 通常对应较低量化风险和较高计算成本，因此可以将配置选择视为一个带性能约束的成本最小化问题。

### 2.2 谱空间局部损失 Proxy

对于线性模块：

```math
Y=XW
```

Metis 分别保留激活和权重的低秩主谱空间：

```math
X=X_{\mathrm{head}}(k_a)+X_{\mathrm{res}}(k_a)
```

```math
W=W_{\mathrm{head}}(k_p)+W_{\mathrm{res}}(k_p)
```

低精度格式 `Q_f` 仅作用于 residual。定义：

```math
\Delta X(k_a)
=
Q_f(X_{\mathrm{res}}(k_a))-X_{\mathrm{res}}(k_a)
```

```math
\Delta W(k_p)
=
Q_f(W_{\mathrm{res}}(k_p))-W_{\mathrm{res}}(k_p)
```

忽略二阶交叉项后，GEMM 输出扰动近似为：

```math
\Delta Y
\approx
\Delta XW+X\Delta W
```

因此，可分别构造 activation 与 weight 两侧的低成本 proxy：

```math
N_i^X(k_a)
=
\frac{
\|Q_f(X_{i,\mathrm{res}}(k_a))-X_{i,\mathrm{res}}(k_a)\|_F^2
}{
\|X_i\|_F^2
}
```

```math
N_i^W(k_p)
=
\frac{
\|Q_f(W_{i,\mathrm{res}}(k_p))-W_{i,\mathrm{res}}(k_p)\|_F^2
}{
\|W_i\|_F^2
}
```

对模型中的目标 GEMM 聚合后得到：

```math
N_X(k_a)
=
\operatorname{Agg}_i N_i^X(k_a)
```

```math
N_W(k_p)
=
\operatorname{Agg}_i N_i^W(k_p)
```

`Agg` 可采用均值、截尾均值或带模块权重的加权聚合。Activation 与 weight proxy 分开保留，避免二者对任务性能的不同影响在求和后丢失。

由于每个矩阵只需执行一次谱分解，即可复用分解结果计算大量候选 rank，因此稠密 proxy landscape 的成本远低于对所有 `(ka,kp)` 组合运行真实下游任务。

### 2.3 任务性能响应面

谱空间 proxy 描述局部 residual 的量化风险，但不能单独替代任务性能。为建立 proxy 与真实任务损失之间的映射，本阶段在 calibration set 上评测少量 `(ka,kp)` 配置，得到稀疏 task anchors：

```math
\mathcal A
=
\{(k_a^{(j)},k_p^{(j)},N_X^{(j)},N_W^{(j)},\Delta P^{(j)})\}_{j=1}^{M}
```

基于这些 anchors 拟合性能响应面：

```math
\widehat{\Delta P}(k_a,k_p)
=
f\left(
N_X(k_a),
N_W(k_p),
k_a,
k_p
\right)
```

首轮采用低复杂度、可解释且带正则的模型，例如：

```math
\widehat{\Delta P}
=
\beta_0
+\beta_aN_X(k_a)
+\beta_pN_W(k_p)
+\beta_{ap}N_X(k_a)N_W(k_p)
```

模型同时输出预测不确定性 `σ(ka,kp)`。在性能约束选择中采用保守判据：

```math
\widehat{\Delta P}(k_a,k_p)
+\lambda\sigma(k_a,k_p)
\le
\epsilon_f
```

该判据用于降低预测误差导致最终配置违反 SOW 性能要求的风险。

### 2.4 Latency 与成本函数

对于全局统一 `(ka,kp)` 配置，latency 可通过目标硬件上的 kernel profiling 得到：

```math
T(k_a,k_p)
=
T_0+T_a(k_a)+T_p(k_p)
```

在 rank 范围较小时，可使用线性或分段线性近似：

```math
T(k_a,k_p)
\approx
T_0+c_ak_a+c_pk_p
```

如果本轮主要验证搜索策略，可先使用 rank cost：

```math
C(k_a,k_p)=c_ak_a+c_pk_p
```

最终配置选择仍以目标硬件上的实测 latency 为准。

## 3. 稀疏标定与迭代搜索方法

### 3.1 计算稠密 Proxy Landscape

首先在 calibration set 上采集模型激活，并对权重和激活执行谱空间分析。候选 rank 可以比真实 task sweep 更密，例如：

```math
K=\{0,5,10,15,20,\ldots,100\}
```

对全部候选 `(ka,kp)` 计算：

- `activation_proxy = N_X(ka)`；
- `weight_proxy = N_W(kp)`；
- `proxy_score`；
- `rank_cost` 或预测 latency。

该阶段不运行完整下游任务，因此可以覆盖数百甚至更多候选组合。

### 3.2 建立首轮稀疏 Task Anchors

在同一 calibration set 上选择少量配置进行真实任务评测。首轮可采用：

```math
k_a,k_p\in\{20,40,60,80,100\}
```

形成 `5 x 5 = 25` 个 task anchors。首轮采样同时覆盖：

- 低成本区域；
- 中间 rank 区域；
- 高 rank、低风险区域；
- activation 与 weight 不对称配置。

根据模型规模和单点评测成本，也可进一步减少规则网格，并补充 `(0,0)`、`(0,kp_max)`、`(ka_max,0)` 和 `(ka_max,kp_max)` 等边界点。

### 3.3 拟合并选择最低成本候选

使用同时具有 proxy 和真实 task loss 的 anchors 拟合 `ΔP(ka,kp)` 响应面，然后在全部稠密 proxy 候选中求解：

```math
(\hat k_a,\hat k_p)
=
\arg\min_{k_a,k_p}
T(k_a,k_p)
```

满足：

```math
\widehat{\Delta P}(k_a,k_p)
+\lambda\sigma(k_a,k_p)
\le
\epsilon_f
```

当没有候选满足约束时，算法报告当前格式、模型和 rank 范围内未发现可行点，并优先在最高 rank 与最低预测损失区域继续验证，而不是强制输出不满足要求的配置。

### 3.4 约束边界主动采样

首轮拟合结果主要用于定位性能约束边界，而不是直接作为最终结论。第二轮真实评测集中在以下位置：

1. 当前预测的最低成本可行点；
2. 与该点相邻、成本更低但接近约束边界的配置；
3. 相同或相近成本下，`ka/kp` 分配不同的非对称配置；
4. 模型预测不确定性较高且可能改变最优解的配置。

例如，当预测候选为 `(ka=40,kp=20)` 时，可优先补充：

```math
(30,20),\ (40,10),\ (40,30),\ (50,20),\ (30,30)
```

新结果加入 anchor 集后重新拟合性能响应面，并再次执行约束选择。

### 3.5 迭代停止条件

当满足以下条件时结束 calibration 搜索：

- 最低成本候选在连续两轮中保持稳定；
- 候选配置已在 calibration task evaluation 中满足性能约束；
- 所有相邻的更低成本配置均不满足约束，或其保守预测无法满足约束；
- 继续采样预计不会改变当前最低成本可行解。

### 3.6 Held-out 与完整数据集确认

搜索过程仅使用 calibration set。最终验证使用与 calibration set 隔离的数据：

1. 在 held-out set 上评测最终候选；
2. 同时评测一个相邻的更低成本点，确认约束边界；
3. 评测高精度 baseline 与必要的高 rank 对照点；
4. 最终在完整任务数据集上确认 SOW 指标。

若最终候选在 held-out 或完整数据集上未满足约束，则将该点作为新的边界信息，继续向更高 rank 或更低 proxy 风险区域进行一轮局部搜索。

## 4. 方法有效性评价

本阶段将完整 `ka/kp` sweep 作为开发阶段 oracle，用于评价稀疏搜索方法，而不作为实际搜索流程的一部分。重点报告以下指标：

### 4.1 性能约束满足情况

- HIF8 最终配置是否满足 performance loss `< 0.5%`；
- HIF4 最终配置是否满足 performance loss `< 1%`；
- calibration、held-out 与完整数据集结论是否一致。

### 4.2 搜索效率

```math
\text{Evaluation Reduction}
=
1-
\frac{N_{\text{evaluated}}}{N_{\text{full sweep}}}
```

报告真实 task evaluation 数量，包括首轮 anchors、主动采样点和最终验证点。

### 4.3 与 Oracle 最优点的差距

设完整 sweep 中的最低成本可行点为 `C_oracle`，搜索方法得到的配置为 `C_search`，报告：

```math
\text{Cost Gap}
=
\frac{T(C_{\text{search}})-T(C_{\text{oracle}})}{T(C_{\text{oracle}})}
```

同时报告搜索结果是否命中 oracle 最优点，或是否落在同一最低成本 Pareto 区域。

### 4.4 Proxy 与真实性能的一致性

报告以下关系：

- `activation_proxy` 与 task performance drop；
- `weight_proxy` 与 task performance drop；
- 综合预测值与 task performance drop；
- Pearson / Spearman correlation；
- 回归误差及性能约束边界附近的预测误差。

## 5. 8B 实验结果（待补充）

本章用于收录 8B 模型上的 HIF8 与 HIF4 结果。正式结果将在完成稠密 proxy、稀疏 task anchors、主动采样和完整数据集验证后补充。

### 5.1 实验设置

| 项目 | HIF8 | HIF4 |
|---|---:|---:|
| 模型 | 待补充 | 待补充 |
| Baseline | 待补充 | 待补充 |
| Calibration set | 待补充 | 待补充 |
| Held-out set | 待补充 | 待补充 |
| 完整评测集 | 待补充 | 待补充 |
| 候选 rank 范围 | 待补充 | 待补充 |
| 首轮 task anchors 数量 | 待补充 | 待补充 |
| 主动采样数量 | 待补充 | 待补充 |
| 性能损失约束 | `< 0.5%` | `< 1%` |

### 5.2 Proxy 与性能响应面

待补充内容：

- 稠密 `(ka,kp)` proxy heatmap；
- 稀疏 task anchors 在 proxy landscape 上的位置；
- 拟合得到的 performance response surface；
- 预测值与真实 task performance 的相关性和误差。

<!-- 待插入：HIF8 proxy / fitted performance 图 -->

<!-- 待插入：HIF4 proxy / fitted performance 图 -->

### 5.3 搜索过程与最终配置

| 格式 | 首轮候选 | 主动采样轮数 | 最终 `(ka,kp)` | 实测 Performance Drop | Latency / Cost | 是否满足 SOW |
|---|---:|---:|---:|---:|---:|---:|
| HIF8 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| HIF4 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |

### 5.4 与完整 Sweep Oracle 对比

| 格式 | Full Sweep 点数 | 实际评测点数 | Evaluation Reduction | Oracle 最优配置 | 搜索配置 | Cost Gap |
|---|---:|---:|---:|---:|---:|---:|
| HIF8 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |
| HIF4 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 | 待补充 |

### 5.5 阶段结论

待 8B 实验完成后，本节将集中回答：

1. 稀疏标定方法能否在显著减少真实 task evaluation 的情况下找到满足 SOW 的配置；
2. 搜索结果与完整 sweep 的最低成本可行点之间存在多大差距；
3. 谱空间 proxy 对 HIF8 与 HIF4 的性能边界分别具有多强的预测能力；
4. 最终推荐配置在完整数据集上是否稳定满足性能约束。

## 6. 本阶段执行流程

1. 对 8B 模型采集 calibration activations，并计算 HIF8/HIF4 的稠密谱空间 proxy；
2. 选取约 25 个首轮 `(ka,kp)` task anchors；
3. 分别拟合 HIF8 与 HIF4 的 performance response surface；
4. 在性能约束下选择最低预测 latency/cost 的候选；
5. 围绕预测边界执行少量主动采样并迭代更新；
6. 在 held-out set 上确认最终候选及相邻低成本点；
7. 在完整数据集上完成 HIF8 `<0.5%`、HIF4 `<1%` 的最终验证；
8. 使用完整 sweep oracle 评价搜索效率、约束满足率与 cost gap。

本方案将 Adaptive-Metis 的谱空间分析能力与少量真实任务标定结合：谱空间 proxy 用于低成本刻画稠密配置空间，稀疏 task anchors 用于校准 proxy 与实际性能之间的映射，主动采样用于集中确认性能约束边界。由此，可避免对全部 `(ka,kp)` 组合执行昂贵的真实任务评测，并以较低离线成本找到满足 SOW 性能要求的最低成本配置。

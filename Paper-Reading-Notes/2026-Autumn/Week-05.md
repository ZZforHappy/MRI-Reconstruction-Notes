# 第 5 周｜MRI 重建论文阅读笔记

## 论文 1：MPFlow（多模态后验引导 Flow Matching）

**论文**：[MPFlow: Multi-modal Posterior-Guided Flow Matching for Zero-Shot MRI Reconstruction](https://arxiv.org/abs/2603.03710)（Kim 等，MICCAI 2026）  
**代码**：[edshkim98/MPFlow](https://github.com/edshkim98/MPFlow)

**研究问题**：在严重欠采样等不适定 MRI 逆问题中，无条件生成先验即使满足测量一致性，仍可能产生未被观测数据支持的错误解剖结构（extrinsic hallucination）。如何利用同一患者、已配准的高质量辅助模态（如 T1）减少重建幻觉，而不重新训练生成先验？

### 核心创新点

1. **零样本多模态后验引导**：先训练无条件 Rectified Flow；重建时才引入真实测量与辅助 MRI，用梯度引导生成轨迹，不需要针对当前逆问题重新训练生成先验。注意“zero-shot”不是完全不训练任何模型。
2. **PAMRI 跨模态解剖特征**：利用目标/辅助模态各一个 ResNet18 编码器，对配准 Patch（32×32）进行双向 Adaptive InfoNCE 对比学习；使用原始 Patch 的归一化互信息（NMI）设置自适应温度，并辅以 Patch 重建损失，学习跨对比度共享的结构特征，而非直接匹配像素强度。
3. **双重 Guidance + 初始噪声筛选**：采样时联合最小化测量一致性损失和 PAMRI 特征差异，分别针对违背测量的错误与测量零空间中的结构幻觉；从 8 个候选噪声短程预热后选择评分最佳者继续生成。

### 核心方法与公式

**Rectified Flow 生成先验（论文式 3–5）**：

`x_t=(1-t)z+t x_1,\quad z\sim\mathcal N(0,I)`

`\mathcal L_{FM}=\mathbb E\big[\|v_\theta(x_t,t)-(x_1-z)\|_2^2\big]`

**PAMRI 训练（论文式 6–7）**：

`\mathcal L_{SSL}=\mathcal L_{NCE}+\lambda_{rec}\mathcal L_{rec}`

**联合后验引导（论文式 8）**：

`v_{guided}=v_\theta(x_t,t)-\alpha_t\nabla_{x_t}\big[\|\mathcal F(\hat x_{1|t})-y\|_2^2+\lambda_P\mathcal L_P\big]`

其中 `\mathcal L_P` 是预测目标图像与辅助模态在 PAMRI 归一化特征空间中的 MSE。前向算子 `\mathcal F` 在本文表示一般测量/退化算子，不必专指傅里叶变换。

**原文复现注意**：论文 2.5 节写 `\hat x_{1|t}=x_t-(1-t)v_\theta(x_t,t)`，但 2.3 节定义的线性路径和正向速度 `v=x_1-z` 对应的是加号；这两处符号不一致，复现时应核对作者代码。

**推理流程**：候选噪声初始化 → Rectified Flow 预测速度和完整图像 → 计算测量损失及 PAMRI 特征损失 → 对当前状态求梯度并调整速度 → ODE 积分至终点。

### 实验结果

- **数据与任务**：HCP：T2 4× 超分辨率（测试 200 张）；BraTS：FLAIR 8× 模拟 k-space 欠采样（验证集 250 张）；均以完整 T1 为辅助，处理二维轴向切片。
- **设置**：生成先验训练 100k iterations；PAMRI Patch 为 32×32；`\lambda_P=0.1`、`\lambda_{rec}=0.5`；8 个初始噪声候选，预热 `0.2T`。
- **BraTS 8×，Table 1**：MPFlow 100 步 PSNR **31 dB**、SSIM **0.90**、LPIPS **0.08**；500 步为 **31 dB / 0.91 / 0.07**；DynamicDPS 500 步为 **31 dB / 0.89 / 0.09**。
- **消融（Table 2，BraTS，100 步）**：Base Dice 0.704；+PAMRI 0.741；+Noise 0.735；完整 MPFlow **0.743**。表明 PAMRI 主要改善解剖结构相关指标，噪声筛选更明显改善测量残差。
- **计算口径**：名义“100 步”并不等于总共只调用网络 100 次；8 候选噪声预热也产生额外计算。

### 局限性与思考

1. **依赖额外完整、配准的辅助模态**：若没有 T1，或者模态间错位/出现病灶差异，PAMRI 的引导未必可靠；需专门测试错误引导风险。
2. **不等于真实多线圈原始数据验证**：论文的 BraTS 任务是从已有图像构造的模拟 k-space 欠采样，未证明在原始复数多线圈 fastMRI、ESPIRiT/SENSE 条件下具有相同效果。
3. **幻觉改善不等于彻底消除**：PSNR/SSIM、SHAFE 和分割 Dice 都是有限的评价证据，不能直接代表临床安全性。
4. **零样本与效率的边界**：不必为测试逆问题重新训练先验，但 PAMRI 本身需要配对多模态预训练；预热筛选增加采样开销。

### 我的理解

**核心不是提出新的 Flow Matching，而是“无条件 Rectified Flow 先验 + 跨模态结构特征 PAMRI + 测量/特征联合后验引导”。** 论文的主要目标是利用同一患者的其他 MRI 模态减少生成式重建的错误结构。对单模态多线圈欠采样研究而言，值得继续追问：没有额外 T1 时，能否仅依靠真实已采集的 k-space、ACS 和线圈关系提供可靠的引导信息？这是研究问题，不是本文已实现的方案。

---

## 论文 2：ReCoFlow-MRI（GNL 感知的少步条件流匹配）

**论文**：[ReCoFlow-MRI: Few-Step Conditional Flow Matching for GNL-Induced Nonlinear MRI Reconstruction](https://arxiv.org/abs/2611.27590)（Qidi Luo 等，2026-11-20 arXiv 预印本）  
**全文**：[PDF](https://arxiv.org/pdf/2611.27590)  
**发表状态**：论文使用 IEEE Transactions on Biomedical Engineering 的排版格式，但目前不能据此确认正式接收。

**研究问题**：欠采样 MRI 存在混叠伪影，梯度磁场非线性（GNL）还会造成空间编码偏差和几何畸变。如何在考虑真实非线性编码的同时，以很少的 Flow Matching 推理步数得到准确重建？

### 核心创新点

1. **GNL-aware 条件构造**：通过网格模体标定 GNL 位移场 `\Delta L`，将其融入非线性 MRI 编码算子 `A_{\Delta L}`，再由其伴随运算构造畸变感知的 Zero-filled 条件图像 `c`。GNL 标定与 NUFFT 均为已有技术，本文贡献在于将其用于条件 Flow Matching 重建。
2. **条件速度场重建**：网络接受当前生成状态、时间与条件图像 `c`，从高斯噪声沿直线概率路径学习到完整 MRI 的速度场；不依赖另一套 T1/FLAIR 辅助模态。
3. **五步终点监督微调（Endpoint Fine-tuning）**：先进行标准速度匹配预训练，再对五步 Euler 积分后的最终图像施加 L1 损失；只通过最后一步反向传播，以较低成本改善有限步重建质量。

### 核心方法与公式

**GNL 感知测量模型**：

```math
y=M A_{\Delta L}x,\qquad
(A_{\Delta L})_{k,r}=\exp\{-i2\pi k\cdot[r+\Delta L(r)]\}
```

**条件图像（GNL-aware zero-filled）**：

```math
c=A_{\Delta L}^{H}y
```

其中伴随算子不等于严格的逆算子，`c` 仍可能存在欠采样伪影和残余误差。

**Conditional Flow Matching 预训练**：

```math
x_t=(1-t)x_0+t x_1,\qquad x_0\sim\mathcal N(0,I)
```

```math
\mathcal L_{CFM}=\mathbb E\left[\left\|v_\theta(x_t,t\mid c)-(x_1-x_0)\right\|_2^2\right]
```

**五步 Euler 推理与终点微调**：

```math
z_{k+1}=z_k+\tfrac15v_\theta(z_k,k/5\mid c),\quad k=0,\ldots,4
```

```math
\mathcal L_{pix}=\mathbb E[\|z_5-x_1\|_1]
```

训练时前四步的状态停止梯度传播，只让最后一步的网络调用参与终点损失的反向传播。

**完整流程**：GNL 位移场标定 → GNL-aware NUFFT 伴随构造 `c` → CFM 预训练 → 五步终点微调 → 五步 Euler 推理。

### 实验结论

- **模拟数据**：在 fastMRI 脑部数据上模拟欠采样与 GNL，论文报告五步 ReCoFlow-MRI 相对所比较的方法（包括 128 步扩散采样）取得较优 PSNR/SSIM。
- **实测场景**：在 MRI-Linac 模体与志愿者数据上进行验证。网格模体的最大几何误差从 **24.95 mm** 降至 **2.12 mm**；3718 个标记点中 **98.39%** 的误差低于 1 mm。
- **消融意义**：论文报告终点微调能进一步提升有限步重建指标，但需要与匹配算力、相同物理条件的基线公平比较。

### 局限性与个人思考

1. **组合式创新为主**：GNL 校准、NUFFT、CFM 均已有研究。更有针对性的设计是针对五步终点误差的微调，而非提出全新 Flow Matching 理论。
2. **GNL 物理约束主要进入条件输入**：后续 ODE 更新没有显式逐步执行 `MA_{\Delta L}\hat x=y` 的 Data Consistency，因此物理一致性不由生成结构严格保证。
3. **伴随图像并非完全去畸变结果**：`c=A_{\Delta L}^{H}y` 只是利用标定位移场进行物理感知的初始重建，剩余误差交由网络学习处理。
4. **应用场景依赖 GNL 标定**：如果只研究普通 fastMRI 欠采样，直接移植方法可能变成“物理预处理 + 条件生成”的简单拼接。
5. **研究启发**：更值得研究的是有限步 Flow 的训练目标与实际重建误差之间的差距，以及如何在少步生成时显式保留真实多线圈 k-space 测量一致性；这些是可探索方向，并非本文已证明的贡献。

### 我的理解

**ReCoFlow-MRI = GNL-aware Zero-filled 条件 + Conditional Flow Matching + 五步 Endpoint Fine-tuning。** 和 MPFlow 的“无条件先验 + 辅助模态后验引导”不同，它把欠采样物理信息直接送入速度网络训练。论文更偏向具有实际 MRI-Linac 应用价值的组合式工作；算法层面最值得学习的是**从速度匹配转向有限步终点监督**。

---

## 论文 3：FLAT（用 Flow-Aligned Training 训练 MRI 深度展开网络）

**论文**：[FLAT: FLow-Aligned Training of Unrolled Networks for MRI Reconstruction](https://arxiv.org/abs/2512.03020)（Kehan Qi 等；2025 年 12 月首次公开，2026 年更新）  
**全文**：[arXiv PDF](https://arxiv.org/pdf/2512.03020v3)

**研究问题**：传统 MRI 深度展开网络由多个 Cascade 逐步重建，通常重视最终输出，但中间更新可能出现不稳定、质量反复波动。如何借鉴连续时间 Flow 的轨迹概念，约束中间迭代并改善重建？

### 核心创新点

1. **展开网络与条件流 ODE 的数学联系（Proposition III.1）**：在特定一阶梯度更新、后验能量及参数对应关系下，可把一个 Cascade 解释为连续动力系统的一次 Euler 离散步。这个命题不是“任意展开网络都等价于精确的生成式 Flow Matching”。
2. **ODE-consistent Scheduling and Parameters**：为各级 Cascade 分配时间节点 `t_k` 和间隔 `\Delta t_k`，按照连续 ODE 的离散关系约束更新步长/参数，而不让各级参数完全无关地自由变化。
3. **Velocity Alignment（最核心的训练设计）**：以 Zero-filled 与完整参考图像之间的线性插值作为理想中间轨迹，监督各级 Cascade 的更新速度，使中间输出更靠近对应时间点的参考状态。

### 核心公式与理解

**传统一阶展开更新（示意，正向编号）**：

```math
x^{(k+1)}=x^{(k)}-\eta_kA^H(Ax^{(k)}-y)+\eta_k\mu\Phi_k(x^{(k)})
```

**从后验能量到连续更新（理论解释）**：

```math
E(x;y)=\frac{1}{2\sigma^2}\|Ax-y\|_2^2-\log p(x)
```

```math
\frac{dx_t}{dt}=-\lambda(t)\nabla_x E(x_t;y)
```

这是特定能量梯度动力系统与一阶展开网络的形式对应；并不能单凭此式认定算法在精确采样后验分布。

**线性插值参考轨迹**：

```math
x_t^*=(1-t)x_{\mathrm{ZF}}+t x_{\mathrm{GT}}
```

**参考速度与实际 Cascade 速度（便于理解的离散表示）**：

```math
v_k^{\mathrm{target}}=\frac{x^*_{t_{k+1}}-x^{(k)}}{\Delta t_k},
\qquad
v_k^{\mathrm{pred}}=\frac{x^{(k+1)}-x^{(k)}}{\Delta t_k}
```

**速度对齐的直观等价解释**：采用相同范数、同一当前状态的定义时，

```math
\|v_k^{\mathrm{target}}-v_k^{\mathrm{pred}}\|
=\frac{1}{\Delta t_k}\|x^{(k+1)}-x^*_{t_{k+1}}\|
```

因此这一速度对齐项可解释为**按时间步长加权的中间图像监督**。上面是帮助理解的等价化简，完整训练损失的系数、求和和监督设定应以论文版本为准。

### 方法流程

**Zero-filled → 多个 CNN/物理 DC Cascade → 重建图像**。FLAT 保留展开式重建主体，通过时间调度和中间轨迹监督改进训练，**并未另外训练从高斯噪声生成 MRI 的标准 Flow Matching 速度网络**。

### 实验与局限

- 作者在三个 MRI 数据集上报告 FLAT 改善重建质量和迭代稳定性，并将其与传统展开网络和生成式方法进行比较。
- **需关注的关键消融**：自由学习参数 vs. ODE 一致时间调度；仅最终损失 vs. 普通 Deep Supervision vs. 线性插值中间监督/Velocity Alignment。若差异主要来自中间监督，Flow 理论框架对性能提升的必要性仍需论证。
- **方法边界**：将一阶迭代写成 ODE 离散本身不是全新数学结论；选择线性中间轨迹也是一种监督假设，而非真实 MRI 重建必须遵循的最优轨迹。

### 我的理解

**FLAT 本质上是一种受 Flow Matching/连续概率流启发的深度展开网络训练策略，不是独立的生成式 Flow Matching MRI 重建模型。** 它最值得记住的是：用线性插值参考状态约束每个 Cascade 的迭代过程，并为时间步长建立一致性调度。创新性评估时，重点比较 Velocity Alignment 与普通加权中间层监督究竟有多大区别。

---

## 本周总结

- **MPFlow**：通过配准的辅助模态引入额外解剖信息，帮助无条件 Flow Matching 在 MRI 逆问题中减少生成幻觉。
- **ReCoFlow-MRI**：GNL 感知的物理条件图像 + 条件 Flow Matching + 五步终点微调，主要解决少步生成重建及空间畸变问题。
- **FLAT**：把展开网络与特定连续流联系起来，用时间调度和线性参考轨迹监督 Cascade；本质仍为级联重建，而非独立生成式 Flow Matching。
- **三篇对比**：MPFlow 侧重推理期后验引导；ReCoFlow-MRI 直接训练条件速度网络并做终点微调；FLAT 主要用 Flow 思想训练传统深度展开网络。
- **后续阅读重点**：如何让少步 Flow 在真实多线圈原始 k-space 上保持显式数据一致性；终点微调是否真正改善泛化。

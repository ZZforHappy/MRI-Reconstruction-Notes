# 第 4 周｜MRI 重建论文阅读笔记

## 论文 1：Guided MRI Reconstruction via Schrödinger Bridge（I²SB-Inversion）

**论文**：Yue Wang 等，*IEEE Transactions on Medical Imaging*，2026。DOI：[10.1109/TMI.2026.3725165](https://doi.org/10.1109/TMI.2026.3725165)  
**代码**：[zhyjSIAT/I2SB-Inversion](https://github.com/zhyjSIAT/I2SB-Inversion)

**研究问题**：在多对比度引导 MRI 重建中，用同一受试者的完整 T1 图像帮助重建欠采样 T2 图像。已有方法通常采用特征融合，难以显式利用跨对比度结构对应；另外 T1 与 T2 可能因运动或对比度差异产生错位，错误引导会引入伪结构。

### 核心创新点

1. **基于薛定谔桥的跨对比度引导重建（I²SB-Recon）**：将完整 T1 作为引导端点、目标 T2 作为目标端点，在图像域学习 T1 → T2 的概率映射，而不是从纯高斯噪声开始生成。I²SB 本身是已有模型，本文创新在于将其用于测量约束下的跨对比度 MRI 重建。
2. **桥采样中融入 k-space 数据一致性**：每步由残差网络预测目标 T2，再用共轭梯度（CG）结合真实欠采样 k-space 修正预测，之后按桥的后验分布继续采样，使生成先验受 MRI 测量约束。
3. **Inversion 引导校正策略（核心创新）**：先用原始 T1 重建初步 T2；再由初步 T2 沿确定性桥轨迹正向反演，生成与目标结构更匹配的新引导变量；最后以新引导重新重建 T2。用于减轻 T1/T2 错位及引导中错误结构带来的伪影。

### 方法流程与核心公式

**MRI 前向模型（式 9）**：

$`y=Ax+\xi,\qquad A=MFC`$

其中 $M$ 为欠采样掩码，$F$ 为傅里叶变换，$C$ 为线圈敏感度。

**残差网络预测目标图像（式 11）**：

$`x_0^\epsilon=x_n-\sigma_n\epsilon_\theta(x_n,n)`$

**CG 数据一致性（式 12–13）**：

$`\bar x_0^\epsilon=\arg\min_x\left[\|y-Ax\|_2^2+\lambda\|x-x_0^\epsilon\|_2^2\right]`$

**Inversion 更新（式 20）**：

$`x_{n+1}=x_n+\frac{\alpha_n^2\sigma_{n+1}^2}{\sigma_n(\alpha_n^2+\sigma_n^2)}\epsilon_\theta(x_n,n)`$

Inversion 使用确定性近似（$\eta=0$），沿桥从目标 T2 端向引导端移动；它并不是普通的随机正向加噪过程。

**完整推理（Algorithm 3）**：

$`b\xrightarrow{\mathrm{I^2SB+DC}}x'_0\xrightarrow{\mathrm{Inversion}}\hat b\xrightarrow{\mathrm{I^2SB+DC}}\hat x_0`$

即 **第一次反向重建 → 正向 Inversion → 第二次反向重建**。整个流程使用同一训练好的残差预测网络，而不是三个独立模型。

### 实验结果

- **数据**：作者自行采集的配对 T1/T2 MRI。脑部：23 名训练受试者（4738 对切片）、6 名测试受试者（1236 对）；膝关节：10 名训练受试者（1600 对）、3 名测试受试者（480 对）。
- **采样设置**：T1 作为完整引导，T2 使用 CAIPI 回顾性欠采样，主要测试 $R=8.66$、$11.38$；ESPIRiT 灵敏度估计。
- **网络**：ADM 预训练初始化的 U-Net，约 5.52 亿参数；实验选择 80 步 SB 采样、每步 7 次 CG 迭代、$\lambda=10^{-5}$。
- **PSNR（Recon → Inversion）**：膝关节 $R=8.66$：38.73 → 39.15 dB；$R=11.38$：37.18 → 37.85 dB。脑部 $R=8.66$：39.06 → 39.18 dB；$R=11.38$：38.15 → 38.37 dB。
- **鲁棒性（Fig. 7–9）**：在人为平移/旋转、层间错位以及引导图像存在虚假结构时，Inversion 能减少错误引导造成的伪影。作者还测试了更高加速因子（Fig. 10）。
- **推理时间（Table III）**：I²SB-Recon 5.55 秒/切片；完整 I²SB-Inversion 14.05 秒/切片，约为前者的 2.53 倍。

### 局限性与值得借鉴的方向

1. **计算量偏大**：完整方案需要两次反向桥采样与一次正向 Inversion；正常配对脑部数据的 PSNR 增益仅约 0.12–0.22 dB，但错位时的鲁棒性收益更明显。
2. **依赖高质量、配对的完整 T1 引导**：当引导本身欠采样或训练配对错位时，还需要进一步验证。
3. **数据独立验证不足**：主要实验基于作者自建数据集，第三方难以直接复算论文指标；这属于可复现性限制，不能据此认定实验造假。
4. **待验证研究假设**：第一次 I²SB-Recon 已获得受 k-space 约束的初步 T2，是否可以用轻量级残差修正网络 + DC，直接纠正误差，替代完整 Inversion 和第二次重建？需在正常对齐、错位、伪结构及相同计算预算下与原方法公平对比。
5. **更有针对性的方案**：结合初步 T2、原始 T1 与 Zero-filled T2，预测局部引导可靠性，减少错位或错误结构的影响；避免仅将一个普通 U-Net 接在后面而缺少明确算法创新。

### 我的理解

**这篇论文的主要创新不是 I²SB 本身，而是“测量约束的跨对比度桥重建 + 通过 Inversion 修正不可靠引导”。** 它最值得借鉴的科研思路是：先发现引导图像可能不可信这一失效场景，再提出针对性的校正机制，最后用错位与伪结构实验验证。后续可重点研究：**能否直接精细化第一次重建结果，以更低计算开销达到与 Inversion 相近的鲁棒性？**

---

## 论文 2：Diffusion Modeling with Domain-conditioned Prior Guidance for Accelerated MRI and qMRI Reconstruction（DiMo）

**论文**：Wanyu Bian 等，*IEEE Transactions on Medical Imaging*（TMI），2025，44(9)：3539–3551。DOI：[10.1109/TMI.2024.3440227](https://doi.org/10.1109/TMI.2024.3440227)  
**全文**：[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11806118/) ｜ [arXiv](https://arxiv.org/abs/2309.00783)

**研究问题**：在 MRI 原生采集域而非仅图像域执行扩散，将测量物理约束融入网络训练及采样。

### 核心创新点

1. **原生域扩散**：Static DiMo 在复数 k-space 中执行 DDPM；Quantitative DiMo 在 qMRI 参数空间扩散。k-space 扩散此前已有研究，贡献是整体框架与任务扩展。
2. **物理先验引导（DC）**：在训练和推理的扩散过程中融入真实欠采样数据，使用时间相关权重约束测量一致性。
3. **扩散内梯度下降（GD）**：利用 MRI 前向模型在扩散步骤中迭代修正数据保真误差，将 DDPM、DC、GD 结合。

### 方法与核心公式

**MRI 前向模型**：

`f=Ax+\epsilon,\quad A=P_{\Omega}FS`

其中 `P_{\Omega}` 为采样算子，`F` 为傅里叶变换，`S` 表示线圈敏感度。

**k-space DDPM 前向扩散**：

`\hat f_t=\sqrt{\bar\alpha_t}\hat f_0+\sqrt{1-\bar\alpha_t}\epsilon`

**噪声预测损失（简写）**：

`\mathcal L=\mathbb E[\|\epsilon-\epsilon_\theta(\hat f_t^{\mathrm{guided}},t)\|_2^2]`

这里 `\hat f_t^{\mathrm{guided}}` 表示经测量约束修正后的带噪状态。

**数据一致性（示意形式）**：

`\hat f_t^{DC}=P_{\Omega}[\lambda_t f+(1-\lambda_t)\hat f_t]+(I-P_{\Omega})\hat f_t`

**梯度下降（式 5）**：

`\hat f^{(k+1)}=\hat f^{(k)}-\eta_k\nabla_{\hat f^{(k)}}\frac12\|AF^{-1}\hat f^{(k)}-f\|_2^2`

**整体流程**：在完整 k-space 上训练去噪网络；每步以 DC/GD 引入物理先验；推理从噪声逐步恢复完整 k-space；最后 IFFT 获得重建图像。

### 实验

- Static DiMo：多线圈膝关节 MRI；25 名受试者（20 名训练，5 名测试）；PD/T2 对比度；4×、5×、6× 欠采样。
- 对比包含传统深度重建、图像域扩散以及已有 k-space 扩散方法 MC-DDPM。
- 框架还拓展到 qMRI 参数空间重建。

### 局限性与个人思考

1. **组合式创新**：DDPM、DC 和 GD 都是成熟技术；“首次将 DDPM 用于 k-space”并不准确。
2. **计算成本高**：多步反向扩散叠加每步迭代 GD，推理效率值得评估。
3. **DC 和 GD 部分功能重叠**：理想单线圈笛卡尔采样且已进行精确硬 DC 时，针对同一数据保真项重复 GD 未必有额外收益。
4. **研究启发**：频率自适应噪声调度、少步采样及复数先验建模值得进一步验证，但须检索相关工作。

### 我的理解

**DiMo = k-space DDPM + 动态 DC + 迭代 GD，并拓展到 qMRI 参数域。** 创新重心在物理先验结合与跨任务验证。后续应该读 **MC-DDPM**，重点比较其与 DiMo 的差别。

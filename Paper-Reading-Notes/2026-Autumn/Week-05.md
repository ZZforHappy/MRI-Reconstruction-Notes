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

## 本周总结

- **MPFlow**：通过配准的辅助模态引入额外解剖信息，帮助无条件 Flow Matching 在 MRI 逆问题中减少生成幻觉。
- **后续阅读重点**：无条件 Flow 的后验引导 vs. 直接训练条件 Flow；跨模态配准鲁棒性；原始多线圈 k-space 验证。

# 第 3 周｜MRI 重建论文阅读笔记

## 论文 1：SSDM-MRI — Highly Undersampled MRI Reconstruction via a Single Posterior Sampling of Diffusion Models

**论文**：Jin Liu 等，*IEEE Transactions on Medical Imaging*，2026。DOI：[10.1109/TMI.2026.3654585](https://doi.org/10.1109/TMI.2026.3654585)

**研究问题**：扩散模型能够提高高加速欠采样 MRI 的重建质量，但传统方法需要大量反向采样步骤，推理耗时较长。本文希望在尽量保持图像细节与数据一致性的前提下，将扩散推理压缩为单步。

### 核心创新点

1. **迭代选择性蒸馏（Iterative Selective Distillation）**：借鉴 Progressive Distillation 的“教师两步、学生一步”，但**只蒸馏靠近干净图像的后半段反向路径**，不再对完整的高噪声路径进行蒸馏。每轮将学生初始化为教师的副本，教师固定，训练完成后学生成为下一轮教师。
2. **在蒸馏阶段引入数据一致性约束（DC-guided Distillation）**：除蒸馏的 $`v`$ 预测损失外，对教师、学生的干净图像估计分别执行 k-space 数据一致性，再约束两者的 DC 结果接近，以强化 MRI 测量信息在蒸馏过程中的作用。
3. **捷径反向采样与蒸馏协同（Shortcut Reverse Sampling）**：借鉴 CCDF，不再从纯高斯噪声 $`x_T`$ 开始，而是先对 Zero-filled 重建加噪，直接构造中间状态 $`x'_{t_0}`$。本文取 $`T=128`$、$`t_0=T/8=16`$；捷径使实际反向路径缩短为 16 步，再经过 **4 轮蒸馏：16 → 8 → 4 → 2 → 1**，达到一次反向网络预测完成重建。
4. **面向多线圈 MRI 和 QSM 的应用适配**：多线圈数据先用 Walsh（SVD）法估计灵敏度并合并图像，网络完成初步重建后再展开到各线圈、逐线圈执行 DC、最后重新合并；对三维 QSM 数据沿完全采样的读出方向作 1D FFT，再逐二维切片重建，用重建相位进行后续 QSM 处理。此项主要是**应用扩展**，不是新蒸馏算法。

### 核心公式

**Zero-filled 构造的捷径起点（论文式 6）**：

$`x'_{t_0}=\sqrt{\bar\gamma_{t_0}}x_{\mathrm{cond}}+\sqrt{1-\bar\gamma_{t_0}}\epsilon`$

$`x_{\mathrm{cond}}=A^Hy`$

$`\epsilon\sim\mathcal N(0,I)`$

**蒸馏与数据一致性损失（Algorithm 2，第 18–21 行）**：

$`x_{\mathrm{teacher}}=\mathrm{DC}(x_0'')`$

$`x_{\mathrm{student}}=\mathrm{DC}(x_0''')`$

$`\mathcal L_1=\|\hat v-v\|_2^2`$

$`\mathcal L_2=\|x_{\mathrm{teacher}}-x_{\mathrm{student}}\|_2^2`$

$`\mathcal L_\theta=\mathcal L_1+\mathcal L_2`$

其中 $`\hat v`$ 是由教师两步采样结果构造的蒸馏目标，$`v`$ 是学生网络的预测；$`x_0''`$ 和 $`x_0'''`$ 分别为教师和学生的干净图像估计。**第二项并非学生与真值的直接监督，而是两者经过 DC 后的匹配。**

**投影式数据一致性（单线圈）**：

$`k_{\mathrm{DC}}=(1-M)\odot\mathcal F(\hat x)+y`$

$`x_{\mathrm{DC}}=\mathcal F^{-1}(k_{\mathrm{DC}})`$

已采样位置使用真实 k-space，未采样位置保留网络预测。

### 方法流程

完整采样图像与 Zero-filled 配对数据 → 训练条件扩散教师模型（Algorithm 1，标准 $`v`$-prediction，原始 $`T=128`$ 步）→ 在靠近 $`x_0`$ 的路径进行四轮选择性蒸馏（Algorithm 2）→ 推理时对 Zero-filled 加噪构造 $`x'_{16}`$ → **一次网络预测** → DC → 输出 MRI 重建图像。

### 实验结论与局限

- 论文在 fastMRI 脑部、膝关节及多线圈数据上进行验证，并开展 QSM 加速实验；报告单张 **320×320 脑部切片约 0.45 秒**的重建时间（具体速度依赖实验硬件与实现）。
- 核心加速来自“**缩短采样起点 + 压缩剩余步骤**”，而非减少 U-Net 模型大小。
- $`t_0=T/8`$ 是经验设定；单步模型仍面临蒸馏误差与重建质量的权衡。多线圈处理采用先合并、后逐线圈 DC 的流程，未直接在扩散网络内部联合建模全部线圈。

### 我的理解

**Algorithm 1 基本是常规的条件扩散预训练；论文的核心创新集中在 Algorithm 2 的选择性蒸馏与 DC 蒸馏损失，以及配套的 Shortcut 采样。** 其创新重点是改变扩散模型的**训练与采样策略**，使传统耗时的扩散重建变成近似 U-Net 级推理速度的单步重建。

**可进一步思考的方向**：分析捷径起点 $`t_0`$ 的自适应选择、蒸馏误差在不同加速因子下的变化，以及是否能将多线圈灵敏度算子更深地融入单步扩散网络，而非只作为后处理 DC。

---

## 论文 2：FDB — Learning Fourier-Constrained Diffusion Bridges for MRI Reconstruction

**论文**：M. Usama Mirza 等，*IEEE Transactions on Medical Imaging*，2026。代码：[icon-lab/FDB](https://github.com/icon-lab/FDB)

**研究问题**：传统扩散 MRI 重建主要学习从高斯噪声到清晰图像的去噪过程，与实际 MRI 欠采样产生的混叠伪影不一致。FDB 希望让扩散过程直接对应 k-space 欠采样，并减少反向恢复时的误差累积。

### 核心创新点

1. **傅里叶约束扩散桥（Fourier-Constrained Diffusion Bridge）**：不添加高斯噪声，而是在正向过程中随机删除 k-space 频率点，连接全采样图像与具有实际欠采样程度的图像，直接学习去混叠（Dealiasing）。
2. **二维外围到中心的渐进退化**：采用 2D 逐点随机删除，优先删除外围高频，尽量保留低频/中频；每步删除的位置不重复，使退化逐步加重且比整行删除更细粒度。
3. **持续校正采样（Continually-Corrected Sampling, CCS）**：反向过程中不仅补回本步缺失的频率，还持续修正此前已经补回的频率，避免普通逐点恢复中误差被永久保留。
4. **校正权重的解析调度**：依据 k-space 保留能量的期望差推导 $w_t$，在反向初期降低校正强度，后期逐步增强；推理时与数据一致性（DC）交替更新。

### 核心公式

**正向退化（累计频率掩码）**：

$`x_t=D_tx_0,\quad D_t=F^{-1}\bar\Lambda_tF,\quad \bar\Lambda_t=\prod_{\tau=1}^{t}\Lambda_\tau`$

其中，$\Lambda_t$ 为当前步二值删除掩码；$D_t$ 在图像域中等价于 **FFT → 掩码删除 → IFFT**。

**恢复网络训练目标（论文式 17）**：

$`\tilde x_0(t)=G_\theta(x_t,t)`$

$`\mathcal L_{\mathrm{FDB}}=\mathbb E_{t,x_0}\left[\|G_\theta(x_t,t)-x_0\|_2^2\right]`$

网络直接预测完整图像 $x_0$，而不是直接预测下一步 $x_{t-1}$。

**持续校正采样 CCS（论文式 21）**：

$`x_{t-1}=x_t+(D_{t-1}-D_t)\tilde x_0(t)+w_tD_t(\tilde x_0(t)-x_t)`$

其中第二项负责**补回本步新增频率**，第三项负责**校正已有频率估计**。

**校正权重（论文式 32）**：

$`w_t=\frac{\mathbb E[\|X_{t-1}\|_2^2-\|X_t\|_2^2]}{\mathbb E[\|X_0\|_2^2-\|X_t\|_2^2]}`$

**推理中的数据一致性（论文式 34）**：

$`\hat x_{t-1}=\dot x_{t-1}+A^H(y-A\dot x_{t-1})`$

### 方法流程

完整 MRI 图像 $x_0$ → 正向随机删除 k-space 频率构造 $x_t$ → 训练 $G_\theta(x_t,t)$ 预测 $x_0$ → 测试时从欠采样数据的 $A^Hy$ 开始 → **CCS 逐步恢复与持续校正** → 每步执行 DC → 输出重建图像。

### 实验与理解

- **数据集**：IXI 脑部 MRI（T1/T2/PD，作为单线圈数据）；fastMRI 脑部 MRI（T1/T2/FLAIR，多线圈数据）。
- **主要设置**：训练起点加速因子 $R'=2$，$T_f=1000$；IXI 训练 40 epochs，fastMRI 训练 10 epochs；测试加速因子包括 $R=4$–$10$。
- **与 DDPM 的区别**：DDPM 主要学习去高斯噪声；FDB 的退化与恢复直接针对欠采样混叠。
- **与普通频率补全的区别**：CCS 不仅补新点，还修正过去恢复过的点，这是该文采样算法的核心。
- **局限与复现关注点**：退化掩码和时间调度需要匹配；推理迭代步数较多；IXI 官方图像及 fastMRI 原始 H5 还需要按代码要求预处理。

### 一句话总结

**FDB 用符合 MRI 欠采样物理过程的“删频率—补频率”代替传统加噪—去噪扩散，并用 CCS 持续校正已恢复的频率，结合 DC 完成重建。**
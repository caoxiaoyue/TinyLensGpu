# Vegetti 系 gravitational imaging / subhalo search 中 source regularization 的处理

## 调研问题与结论

**问题：**在 smooth-vs-subhalo 的贝叶斯证据比较中，像素化 source 的 regularization strength / hyperparameter 是固定、重新优化（type-II / empirical-Bayes evidence maximization），还是与其它参数一起在 posterior 中采样并边缘化？

**结论（先给最短答案）：**经典 Vegetti 路线不是把 smooth 模型得到的 source regularization strength 固定后直接沿用到 subhalo 模型。

1. **求 MAP / 建立起点时**，`λ_s` 与宏观透镜参数交替优化；对 clumpy/subhalo 模型还会重新优化。这是 profile / type-II evidence maximization 层面。
2. **最终 smooth-vs-parametric-subhalo Bayes factor 时**，2009 方法论文、2010 和 2012 实际工作都把 `λ_s` 当成有先验（通常 `log λ_s` 上均匀）的非线性超参数，和 lens/subhalo 非线性参数一起用 Nested Sampling / MultiNest **积分边缘化**。所以最终证据不是简单的 `Z(λ_s=λ̂_smooth)`，也不只是各模型在 `λ̂` 处的 profile evidence。
3. **source pixel amplitudes `s`** 不是作为成百上千维参数交给 MultiNest 逐像素采样；在给定 `(η, λ_s, R_s)` 时通过线性系统求 MAP，并因高斯 likelihood + 二次 regularization 可在条件证据中**解析积分掉**。
4. **regularization form `R_s/H_s` 与 source grid/pixelization** 通常作为离散模型选择：用 evidence 选择一种形式/网格，随后 smooth 与 subhalo 比较保持同一形式和同一网格规则。论文没有把这些离散选择在同一个 posterior 内做模型平均。
5. 需另行区分**像素化 potential-correction 的发现/定位阶段**：早期工作常固定或手工扫描 potential-regularization `λ_{δψ}`；这不等于最终 parametric smooth-vs-subhalo evidence 对 source `λ_s` 的处理。
6. 后续样本工作的**sensitivity-map 计算**有时为计算量而固定 macro/subhalo 参数，仅对 source（含 `λ_s`）边缘化。这是 sensitivity approximation，不应倒推为经典检测 Bayes factor 固定 `λ_s`。

因此，如果 M3/M4 是要复现经典 Vegetti 的最终 smooth-vs-subhalo 证据比较，那么把 M2 的 `log_lambda_reg` / adaptive-regularization hyperparameter 固定过去，只能叫**条件证据或计算近似**，不能叫对 source-regularization uncertainty 的完整 marginalization；也不同于经典流程中“每个模型先重优化 `λ_s`，最终再积分 `λ_s`”。

---

## 先统一五类量与三层推断

记数据为 `d`，线性 source pixels 为 `s`，非线性 lens/subhalo 参数为 `η`，source regularization strength 为 `λ_s`，regularization operator/form 为 `R_s`（或 `H_s`），模型族为 `M`。

### (1) 线性 source pixels

对固定的 `(η, λ_s, R_s, M)`，求解

\[
\hat s=\arg\min_s\left[E_D(d\mid s,\eta,M)+\lambda_s E_S(s\mid R_s)\right].
\]

在高斯噪声和二次正则下，条件证据

\[
\mathcal L(\eta,\lambda_s;M,R_s)
=P(d\mid\eta,\lambda_s,M,R_s)
=\int ds\,P(d\mid s,\eta,M)P(s\mid\lambda_s,R_s)
\]

可解析计算。这里“source 被 marginalize”不表示逐像素 MCMC/Nested Sampling。

### (2) source regularization strength

- **MAP/type-II 层：**寻找 `λ̂_s`（通常与 `η` 交替优化），用于最佳模型和初始化。
- **最终 model evidence 层：**

\[
Z(M,R_s)=P(d\mid M,R_s)
=\int d\lambda_s\,d\eta\;\mathcal L(\eta,\lambda_s;M,R_s)P(\lambda_s,\eta).
\]

经典工作的最终 Bayes factor 使用后一式；不能把“先优化了 `λ_s`”误读成“最终固定 `λ_s`”。

### (3) regularization form / grid

`R_s`、source grid resolution 或取样规则通常由若干离散候选的 evidence 排名决定。最终 smooth/subhalo 比较通常条件于已经选定的同一 `R_s` 和 grid；没有对离散 form/grid 做完整模型平均。

### (4) lens/subhalo 非线性参数

MAP 阶段优化；最终证据阶段由 Nested Sampling / MultiNest 在给定先验体积中积分。subhalo mass 和 position 是 clumpy 模型新增的非线性参数。

### (5) “最优超参数条件证据”还是“完整边缘化”

同一篇论文常同时出现：

- 最佳拟合/MAP：`η̂, λ̂_s`；
- 最终模型证据：对 `η, λ_s` 积分；
- 为 sensitivity map 降成本的近似：固定部分 `η`，只积分 `λ_s`（并隐式积分 `s`）。

下面逐篇区分。

---

## 1. Vegetti & Koopmans (2009)：方法定义已经明确区分 optimize 与 marginalize

**原始论文：**S. Vegetti & L. V. E. Koopmans, *Bayesian strong gravitational-lens modelling on adaptive grids: objective detection of mass substructure in galaxies*, MNRAS 392, 945–963 (2009).  
[arXiv abstract](https://arxiv.org/abs/0805.0201) · [author manuscript PDF](https://arxiv.org/pdf/0805.0201)

### 逐项结论

| 项目 | 2009 处理 |
|---|---|
| 线性 source pixels `s` | 给定 `η, λ` 解线性方程；条件 evidence 中解析积分 |
| source strength `λ_s` | MAP 阶段与 `η` 交替 evidence-maximize；最终模型证据中积分 |
| form/grid | source 采用 Delaunay adaptive grid 和 curvature regularization；可用 evidence 比较 form/grid，但不是同一 posterior 中采样 |
| lens/subhalo nonlinear `η` | MAP 阶段优化；最终 Nested Sampling |
| final evidence | `∫dλ dη`，不是固定在 `λ̂` |

### 精确证据

1. **摘要已直接说清最终误差与 evidence 的 marginalization 范围：**

> “Nested-Sampling … quantify the errors on all non-linear mass model parameters – **marginalized over all source and regularization parameters** – … ranking … in terms of the **marginalized evidence**.”

见摘要，PDF p.1。

2. **线性 source 与解析 source marginalization：**§4 的 first level 给出固定 `λ,η,R` 的 posterior，eqs. (22)–(33)；§4 second level 明写

\[
P(d\mid\lambda,\eta,M_c,R)
=\int dr\,P(d\mid r,\eta,M_c)P(r\mid\lambda,R),\tag{35}
\]

并说因 Gaussian noise 与 quadratic regularization，积分可解析，得到 eqs. (36)–(37)。这里 `r` 包含 source（在 potential-correction 阶段也可含 `δψ`）。

3. **MAP/type-II optimization：**§3.2.2 steps (1)–(3) 先用较大 `λ_s` 优化 `η,s`，再固定最佳 `η` 寻找最佳 `λ_s`，然后重复；§4（eqs. 34–37 后）称这是对 `(η,λ_s)` 的 evidence maximization，并解释实际采用 alternating/line-search，而不是每个 `η` 点套完整 `λ_s` 内循环。原文：

> “we fix the best model … and … **maximize the evidence for the source regularization level** `λ_s` … the alternate loop … gives a faster way to reach the maximum.”

这一步只是在找联合 MAP/初始化，不是最终模型证据。

4. **最终 smooth-vs-clumpy evidence 对 `λ,η` 积分：**§4 third level, eq. (39)：

\[
P(d\mid M_c,R)=\int d\lambda\,d\eta\;
P(d\mid\lambda,\eta,M_c,R)P(\lambda,\eta).\tag{39}
\]

（PDF 文本中的控制字符只是排版抽取问题；公式含义是 `P(λ,η)`。）紧接着写明该高维积分用 Nested Sampling 数值完成。§4.2 又定义 prior-volume element

\[
dX=P(\lambda,\eta)d\lambda d\eta.\tag{41}
\]

§4.2.1 明写 lens parameters、substructure position 和 **logarithm of source regularization** 都有 prior；smooth 与 substructure 模型对 main-lens parameters 和 regularization constant 使用相同 priors。

5. **regularization form/grid：**§3.1.1 表示 gradient/curvature 等 form 可通过 evidence maximization 判断，但本文 source 选 curvature regularization。§5.1 说明 source-grid pixel number 也是 prior/model choice，可用 marginalized evidence 测试。它们是离散模型选择，不在 eq. (39) 中作为连续参数积分。

6. **不要混入 potential-correction regularization：**§3.2.1 明说：

> “We have not attempted to optimize for `λ_{δψ}` …”

并在 discovery 阶段取很大的 `λ_{δψ}` 保持 correction 光滑/线性。最终 analytic smooth-vs-NFW/subhalo comparison 则回到 eq. (39)，source `λ_s` 仍被积分。

---

## 2. Vegetti et al. (2010)：J0946 的 `λ_s` 在 PL 与 PL+PJ 中不同，最终 Bayes factor 又对其边缘化

**原始论文：**S. Vegetti et al., *Detection of a Dark Substructure through Gravitational Imaging*, MNRAS 408, 1969–1981 (2010).  
[arXiv abstract](https://arxiv.org/abs/0910.0760) · [author manuscript PDF](https://arxiv.org/pdf/0910.0760)

### 逐项结论

| 项目 | 2010 处理 |
|---|---|
| 线性 source pixels | 线性反演；条件 evidence 中已积分 |
| source strength `λ_s` | smooth 与 PL+PJ 分别优化；最终 MultiNest 对 `λ_s` 边缘化 |
| form/grid | nominal model 采用 S/N-adaptive curvature；grid point number 用 evidence 选择；form/grid 作为 robustness alternatives |
| lens/subhalo nonlinear | 主透镜、PJ mass/position 优化，最终 MultiNest marginalization |
| final evidence | “total marginalized evidence”；`λ_s` 有 log-flat prior，不是固定值比较 |

### 精确证据

1. **流程声明：**§2 的四步流程写道：

> “determine the non-linear parameters and **level of source regularisation that maximize the Bayesian evidence**”

随后在 parametric PJ 阶段

> “simultaneously optimising for the main lens galaxy and substructure parameters”

最后 PL 与 PL+PJ 用

> “their **total marginalized Bayesian evidences** … marginalized over all variable model parameters”

比较（§2，PDF pp.2–3）。

2. **`λ_s` 没有从 smooth 固定到 subhalo：**Table 1 给出 nominal smooth `(PL)_0` 的 `log(λ_s)=1.152`，而 nominal `(PL+PJ)_0` 为 `2.028`。这直接表明 clumpy 模型使用了自己的最优 regularization，而不是继承 smooth 值。

3. **固定成相同 `λ_s` 的模型只是诊断实验：**§5.2/§6 前的讨论为检验 source–potential covariance，特意构造 `PL_{0,over}`：把 smooth 模型 `λ_s` **固定为最佳 PL+PJ 的值**；Table 1 中确为 `2.028`。论文用它比较 likelihood/residual，强调这是 “over-regularized” robustness diagnostic；正式 `Δlog E=-128.2` 的 smooth baseline 是自由 `λ_s` 的 PL，不是该固定-`λ_s` 模型。

4. **最终 evidence 中 `λ_s` 被 MultiNest 采样/积分：**§6.1 写明

> “The regularization constant has a prior probability which is **logarithmically flat** in a symmetric interval around `λ_{s,b}`.”

§6.2 说明 MultiNest 同时探索 posterior 并计算 marginalized evidence。这里沿用 2009 eq. (39) 的层级；因此 `λ_s` 是 nested-sampling 参数，而 source pixels 已在条件 evidence 中解析积分。

5. **form/grid：**§4.1：source-grid 点数以 Bayesian evidence 选择；nominal model 采用 inverse-image-S/N weighted adaptive curvature regularization，gradient 和 unweighted curvature evidence 更低。§6.5 另跑 non-adaptive 与 gradient regularization 作为系统学检查，没有对这些 form 做 posterior model averaging。

6. **potential correction 不是同一个 `λ` 问题：**§4.2/§5.1 对 `λ_{δψ}=10^7,10^8,10^9` 扫描以确认 density peak 稳健；这属于像素化 potential discovery。不能据此说 parametric PL-vs-PL+PJ evidence 固定了 source `λ_s`。

---

## 3. Vegetti et al. (2012)：Supplementary eq. (7) 明确对 `λ_s` 与 `η` 积分

**原始论文：**S. Vegetti et al., *Gravitational detection of a low-mass dark satellite galaxy at cosmological distance*, Nature 481, 341–343 (2012), 含 Supplementary Information。  
[arXiv abstract](https://arxiv.org/abs/1201.3643) · [author manuscript + Supplementary PDF](https://arxiv.org/pdf/1201.3643) · [Nature article](https://www.nature.com/articles/nature10669)

### 逐项结论

| 项目 | 2012 处理 |
|---|---|
| 线性 source pixels | 像素化 Delaunay source，线性反演/条件积分 |
| source strength `λ_s` | 每个模型 MAP 中优化；最终 evidence 对 `λ_s` 积分，prior 对 `log λ_s` 均匀 |
| form/grid | nominal 采用 gradient regularization；grid resolution 由 evidence；adaptive form 等作为替代模型/系统学测试 |
| lens/subhalo nonlinear | `η`、PJ mass/position 在 MAP 中联合优化，最终在 prior 内积分 |
| final evidence | Supplementary eq. (7) 的 `∫dλ_s dη`，完整 marginalization（在固定 `M,R` 条件下） |

### 精确证据

1. **主文对最终证据的定义：**主文 PDF p.3：

> “Bayesian evidence … probability of the data given a specific model (**marginalized over all parameters**). Computing the marginalized evidence involved **integrating over the multidimensional parameter space** within predefined priors.”

2. **MAP/profile 阶段确实优化 `λ_s`：**Supplementary “Bayesian Lens modelling”，eq. (2) 后写：

> “The source regularisation `λ_s` sets the level of smoothness … and its value was objectively set by maximizing Eq. (2).”

parametric subhalo 段又写：

> “By maximizing the Bayesian posterior probability, Eq. (2), in terms of `η, λ_s`, the substructure mass and substructure position, we found the best substructure models …”

这说明 `λ_s` 随 subhalo model 重新求最佳值，不是固定 smooth 值。

3. **最终 evidence 不是只取上述最大值：**Supplementary “Model comparison”，eq. (7)：

\[
P(d\mid M,R)=\int d\lambda_s\,d\eta\;
P(d\mid\lambda_s,\eta,M,R)P(\lambda_s,\eta).\tag{7}
\]

紧接着明确：

> “The source regularisation constant had a **uniform prior in the logarithmic space**.”

并要求 smooth 与 substructured models 的 priors 相同。主文所报 `Δlog E=-65.10` 因而是 `λ_s` 与 `η` marginalization 后的结果。

4. **posterior 里确有 `λ_s`：**Supplementary Table 2 最后一列给出模型 `M` 的 `log λ_s=-3.012±0.108`，标注为 posterior mean 与标准差；这不是一个被固定、无不确定度的常数。

5. **form/grid：**Supplementary Bayesian Lens modelling 段说 optimal source-grid sampling number 可由 evidence 推断；nominal 使用 gradient regularization。M5 改 grid resolution，M6 使用随 image S/N adaptive 的 source regularization，作为不同模型/系统学检查。最终 eq. (7) 明确是“for a given model `M` and regularisation form `R`”，即 evidence 条件于 `R`，不是对 form 做积分。

---

## 4. Vegetti et al. (2014)：后续样本实践延续“每模型重优化 + 最终积分”；但 sensitivity map 使用降维近似

**原始论文：**S. Vegetti et al., *Inference of the cold dark matter substructure mass function at z = 0.2 using strong gravitational lenses*, MNRAS 442, 2017–2035 (2014).  
[arXiv abstract](https://arxiv.org/abs/1405.3666) · [author manuscript PDF](https://arxiv.org/pdf/1405.3666)

### 检测/模型比较本身

1. **三步 MAP optimization：**§4.1.1：先以大 `λ_s` 优化 lens `η`；再固定 `η` 优化 `P(λ_s|η,d,...)`；最后固定最佳 `λ_s` 再优化 `η`。这是交替求联合 MAP，不是最终证据。

2. **clumpy 模型重新优化：**§4.3：clumpy model 使用与 smooth 相同的 source regularization **form** 和 source-grid pixel number，但 posterior

\[
P(\lambda_s,\eta\mid d,M_c,R_s)
\]

针对 `η={macro lens, M_sub,x_sub,y_sub}` 与 `λ_s` 重新按同一三步流程 maximize。也就是说：**固定的是 form/grid，不是 strength。**

3. **最终 model evidence 对 `λ_s,η` 积分：**§4.4 eq. (10)：

\[
E=P(d\mid M_c,R_s)=\int d\lambda_s\,d\eta\;
P(d\mid\lambda_s,\eta,M_c,R_s)P(\lambda_s,\eta).\tag{10}
\]

MultiNest 计算该积分，`η` 使用 uniform priors，`log λ_s` 使用 uniform prior；smooth 与 clumpy priors 相同。

4. **source pixels：**§4.1.1 eq. (5) 给出给定 `(η,λ_s)` 时 source posterior；`s` 通过直接线性反演取得 most-probable solution，并已包含在条件 normalization/evidence `P(d|λ_s,η,M_c,R_s)` 中。

5. **form/grid：**§4.1.1 对不同 `n_src` 及 curvature/gradient、adaptive/non-adaptive form 重跑 smooth 三步流程，以 evidence 选最佳组合；§4.3 要求 clumpy 与 smooth 使用同一 `n_src,R_s`，保证公平比较。这是 plug-in discrete model selection，不是 form/grid marginalization。

### 必须单列的 sensitivity-map 近似

§6.1 说严格方式应按 eq. (10) 对 macro model、source、`λ_s` 全部边缘化，但每个 image pixel × 多个 subhalo masses 这样做代价过高。实际将 Bayes evidence 降成只对 source regularization constant 的一维积分，**固定 lens 与 substructure parameters**；source surface-brightness 仍通过 eq. (4)/(5) 隐式积分。

原文（§6.1）：

> “reduce the multidimensional integral … to a **one-dimensional integral over the source regularization constant**, while keeping the lens (and substructure) parameters fixed. Because this integral implicitly involve[s] also an integral over the source surface brightness distribution …”

所以 2014 有两种层次：

- 真实候选/模型比较：`η, λ_s` 全积分；
- 大规模 sensitivity map：固定 `η`，但仍积分 `λ_s` 与 source，而不是固定 `λ_s`。

---

## 5. Ritondale et al. (2019)：BELLS 样本的最终 parametric evidence 明确积分 `λ_s`、macro、light、halo mass/position

**原始论文：**E. Ritondale et al., *Low-mass halo perturbations in strong gravitational lenses at redshift z≈0.5 are consistent with CDM*, MNRAS 485, 2179–2193 (2019).  
[arXiv abstract](https://arxiv.org/abs/1811.03627) · [author manuscript PDF](https://arxiv.org/pdf/1811.03627)

### 逐项证据

1. **source pixels 与 `λ_s` 都是模型自由量，但处理方式不同：**§3.1 明写其它 free parameters 是每个 source-plane pixel 的 surface brightness 与 regularization level `λ_s`。§3.2 eq. (6) 表明对每组 nonlinear `(η_m,η_l,λ_s)`，线性向量 `r`（source pixels、lens-light 线性归一化、sky）由线性系统求解。

2. **像素化 potential-correction 阶段固定 smooth MAP：**§3.3 明写 `η_m,η_l,λ_s` 在该阶段保持为 §3.2 的 MAP；它用于定位 general potential departures，不是最终 smooth-vs-parametric-halo Bayes factor。

3. **最终 evidence：**§3.5 eq. (12)：

\[
E=P(d\mid M,H_s)=\int
P(d\mid\lambda_s,\eta_m,\eta_l,m,x,M,H_s)
P(\lambda_s,\eta_m,\eta_l,m,x)
\,d\lambda_s\,d\eta_m\,d\eta_l\,dm\,dx.\tag{12}
\]

该积分用 MultiNest；`λ_s` prior 在 logarithmic space 均匀。故最终 parametric smooth/perturbed evidence 并未把 `λ_s` 固定为 smooth MAP。

4. **regularization form：**eq. (12) 条件于 `H_s`；form 不是连续 posterior 参数。文中的检测标准还要求 convergence correction 对 source regularization forms/levels 稳健。

---

## 6. 后续方法澄清：不要把 detection evidence 与 sensitivity shortcut 混为一谈

**原始论文：**G. Despali et al., *Detecting low-mass haloes with strong gravitational lensing I: the effect of data quality and lensing configuration*, MNRAS 510, 2480–2494 (2022).  
[arXiv abstract](https://arxiv.org/abs/2111.08718) · [author manuscript PDF](https://arxiv.org/pdf/2111.08718)

这篇与 Vegetti 合著的后续方法实践把 smooth 与 perturbed evidence 写成显式 source 积分（§2.1 eqs. 3–4），并说：

> “the brightness of each pixel … **as well as** the source regularisation level `λ_s`, are … free parameters”

> “Both integrals are performed with MultiNest … [with] a uniform prior in logarithmic space on … `λ_s`.”

它进一步说明该工作在 sensitivity 计算中同时 marginalize source（含 `λ_s`）和 main-lens parameters；Appendix A 则准确回顾 2014 的 shortcut：宏观模型固定在 most-probable value，只 marginalize source light 与 its regularisation level。这个回顾与 2014 §6.1 一致。

注意：这不否定 2014/2019 对实际 smooth-vs-clumpy **detection evidence** 写出的全积分；它针对的是逐像素 sensitivity-map 计算中的 computational approximation。

---

## 汇总矩阵

| 论文/阶段 | source pixels `s` | `λ_s` MAP | `λ_s` final evidence | form/grid | nonlinear lens/subhalo | evidence 类型 |
|---|---|---|---|---|---|---|
| Vegetti & Koopmans 2009 | 线性解；解析积分 | 与 `η` 交替 type-II optimize | Nested Sampling 积分 | curvature + adaptive Delaunay；离散 evidence 选择 | NS marginalization | `∫dλ_s dη` |
| Vegetti et al. 2010 J0946 | 同上 | PL 与 PL+PJ 分别优化；Table 1 值不同 | log-flat prior 下 MultiNest 积分 | nominal adaptive curvature；替代 form 作 robustness | macro + PJ mass/pos marginalize | total marginalized evidence |
| Vegetti et al. 2012 B1938 | 同上 | 每个 subhalo model 联合优化 `η,λ_s,M,x` | Supplement eq. (7) 积分；Table 2 有 `λ_s` posterior width | nominal gradient；grid/form 为离散选择 | 全部积分 | `∫dλ_s dη` |
| Vegetti et al. 2014 detection | 同上 | smooth/clumpy 各自三步重优化 | MultiNest eq. (10) 积分 | smooth 选定后 clumpy 保持同 form/grid | 全部积分 | full marginalized evidence |
| Vegetti et al. 2014 sensitivity | source 隐式积分 | — | **仅积分 `λ_s`** | 固定已选 form/grid | macro/subhalo 固定 | computational approximation |
| Ritondale et al. 2019 detection | 线性系统求解；条件 evidence 隐式积分 | smooth MAP 用于 potential map | MultiNest eq. (12) 积分 | 条件于 `H_s` | `η_m,η_l,m,x` 全积分 | full marginalized evidence |
| Despali et al. 2022 sensitivity | 积分 | — | MultiNest 积分 | 条件于选定形式 | main lens 也积分 | 比 2014 sensitivity 更完整 |

---

## 对当前 M2 → M3/M4 设计的直接含义

若 `log_lambda_reg` 和 `adaptive_reg_rho` 都控制 source prior/regularization：

1. **固定到 M2 posterior median：**得到的是
   \[
   Z_{M3/M4}\bigl(\lambda=\lambda_{M2,\mathrm{med}},\rho=\rho_{M2,\mathrm{med}}\bigr),
   \]
   是条件模型比较；不等于 Vegetti 经典 final evidence。
2. **每个 M3/M4 重新 evidence-optimize：**更接近 Vegetti 的 MAP/type-II 阶段，但仍是 profile/plug-in evidence；尚未传播 hyperparameter prior volume。
3. **在每个 smooth/subhalo 模型中对它们设共同、明确的 prior，并随 nonlinear lens/subhalo 参数共同采样/积分：**最接近 2009 eq. (39)、2012 eq. (7)、2014 eq. (10)、2019 eq. (12) 的最终证据定义。
4. 如果 `adaptive_reg_rho` 改变的不只是强度，而是 regularization operator/form，则经典论文更接近“对若干 form 单独算 evidence 后选择”，而不是在同一 continuous posterior 中自然 marginalize。若要 sample `rho`，应明确它被定义为连续 hyperparameter，且 smooth/subhalo 使用相同 prior；否则 Bayes factor 会混入不对称 prior-volume 效应。

## 一句话判断

**经典 Vegetti smooth-vs-subhalo 证据：source pixels 线性解析边缘化；source regularization strength 先按模型重新优化用于 MAP，再在最终 Nested-Sampling evidence 中边缘化；regularization form/grid 通常固定为 evidence 选出的共同离散方案；lens/subhalo 非线性参数同时边缘化。**

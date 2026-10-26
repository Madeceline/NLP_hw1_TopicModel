# AP-LDA：模型与推断的数学依据

## 1. 定义、联合分布与生成机制

文档作者 a(d) 是给定的单作者索引。K、V、A、D 分别为主题、词汇、作者、文档数量；β、γ、λ 严格为正。

1. 每个主题 k：φ_k ~ Dir(β 1_V)。
2. 每个作者 a：π_a ~ Dir(γ 1_K)。
3. 每个文档 d：θ_d | π_a(d) ~ Dir(λ π_a(d))。
4. 每个词位 n：z_dn ~ Cat(θ_d)。
5. 每个词位 n：w_dn ~ Cat(φ_z_dn)。

\[
p(\Phi,\Pi,\Theta,Z,W\mid a,\beta,\gamma,\lambda)
=\prod_k\operatorname{Dir}(\phi_k;\beta\mathbf1)
\prod_a\operatorname{Dir}(\pi_a;\gamma\mathbf1)
\prod_d\operatorname{Dir}(\theta_d;\lambda\pi_{a(d)})
\prod_{dn}\theta_{d,z_{dn}}\phi_{z_{dn},w_{dn}}.
\]

标准 LDA 删除作者层，改用 θ_d ~ Dir(α 1_K)。公平浓度匹配设 α=λ/K。

## 2. 理论性质

\[
E[\theta_{dk}\mid\pi_a]=\pi_{ak},\qquad
\operatorname{Var}(\theta_{dk}\mid\pi_a)=\frac{\pi_{ak}(1-\pi_{ak})}{\lambda+1}.
\]

λ 是总浓度。λ 较大时文档紧贴作者偏好；λ 趋向 0 时单篇文档更接近单一主题，期望不变。λ 趋向无穷时 θ_d 收敛至 π_a，但 φ 和 π 仍需从文本估计。

同作者两篇不同文档在给定 π 时独立，边缘上相关。令 q=1/K、c=Kγ，则

\[
\operatorname{Cov}(\theta_{dk},\theta_{ek})=\operatorname{Var}(\pi_{ak})
=\frac{q(1-q)}{c+1},\quad d\ne e,
\]
\[
\operatorname{Var}(\theta_{dk})=
\frac{q(1-q)(c+\lambda+1)}{(c+1)(\lambda+1)},\qquad
\operatorname{Corr}(\theta_{dk},\theta_{ek})=\frac{\lambda+1}{c+\lambda+1}.
\]

因此 λ 不改变边缘跨文档协方差（γ 固定时），但增大相关系数；不能笼统写成“协方差随 λ 增加”。

## 3. 折叠目标与非共轭性

积分掉 Θ、Φ。n_dk、n_kv、n_k· 为文档主题、主题词、主题总计数：

\[
p(Z,W,\Pi)\propto
\prod_k\frac{\Gamma(V\beta)}{\Gamma(n_{k\cdot}+V\beta)}
\prod_v\frac{\Gamma(n_{kv}+\beta)}{\Gamma(\beta)}
\prod_a\operatorname{Dir}(\pi_a;\gamma\mathbf1)
\prod_d\frac{\Gamma(\lambda)}{\Gamma(N_d+\lambda)}
\prod_k\frac{\Gamma(n_{dk}+\lambda\pi_{a(d)k})}{\Gamma(\lambda\pi_{a(d)k})}.
\]

π 的条件分布含 Gamma 函数之比，**不是直接的 Dirichlet 后验**，不能用 γ+词频直接更新。

## 4. CRT 辅助变量给出合法的精确转移核

使用升阶乘恒等式（|s(n,t)| 为第一类无符号 Stirling 数）：

\[
\frac{\Gamma(n+x)}{\Gamma(x)}=x^{\overline n}
=\sum_{t=0}^n|s(n,t)|x^t.
\]

给每个 d,k 引入桌数 t_dk。于是

\[
t_{dk}\mid n_{dk},\pi\sim\operatorname{CRT}(n_{dk},\lambda\pi_{a(d)k}),\quad
t_{dk}=\sum_{i=1}^{n_{dk}}b_i,\quad
b_i\sim\operatorname{Bernoulli}\left(\frac{\lambda\pi_{a(d)k}}{\lambda\pi_{a(d)k}+i-1}\right).
\]

零计数给零桌；非零计数第一位必开桌。将 Γ 比展开后，π 只以幂函数出现，故 **在扩充空间** 中有

\[
\pi_a\mid T,Z\sim\operatorname{Dir}\left(\gamma\mathbf1_K+\sum_{d:a(d)=a}t_d\right).
\]

λ 固定使 Γ(λ)/Γ(N_d+λ) 对 π 为常数，这是上述推导成立的关键。

词主题更新（移除当前词位后的计数用上标 − 表示）：

\[
P(z_{dn}=k\mid Z^{-dn},W,\Pi)\propto
(n_{dk}^{-dn}+\lambda\pi_{a(d)k})
\frac{n_{k,w_{dn}}^{-dn}+\beta}{n_{k\cdot}^{-dn}+V\beta}.
\]

每轮严格执行：①在边缘空间更新全部 Z；②从 T|Z,π 重采桌数；③从 π|T,Z 重采作者偏好；④丢弃旧 T。Z 的更新已经边缘化 T，不能把上一轮 T 当作 Z 更新的约束。此顺序在 (Z,π) 边缘后验上保持不变分布，并通过辅助核处理非共轭作者层。无需 MH、梯度、ELBO 或修改模型。

若需要恢复随机参数：φ_k|Z,W ~ Dir(β+n_k)，θ_d|Z,π ~ Dir(λπ_a+n_d)。代码保存留样时的条件均值并平均，作为 Rao–Blackwell 后验估计；不把优化目标称为 ELBO。

## 5. 新文档与统一预测

训练后固定 φ、已知作者 π 的后验均值，仅用观察半篇词进行局部 Gibbs：P(z_n=k) ∝ (n_k^-+λπ_ak) φ_kw。记录 θ 的后验条件均值。对三个/四个模型均用此同一局部采样器，标准 LDA 用 α=λ/K。

新作者：从 Dir(γ1) 初始化其 π，使用相同 CRT 核在单篇观察文本内联合推断 π、Z；其他测试文档不参加该推断。没有作者标识时采用此相同策略。空文本返回对称均值。已知作者不利用测试文本重新训练全局 π。

文档完成分数：随机选择一部分**词位**作为观察集 O，其余 H 仅评分：

\[
\ell=\frac{1}{\sum_d|H_d|}\sum_d\sum_{w\in H_d}
\log\left(\sum_k\widehat\theta_{dk}(O_d)\widehat\phi_{kw}\right),
\qquad\operatorname{PPL}=\exp(-\ell).
\]

这是全局参数后验均值的 **plug-in 单词边缘预测分数**，不是完整贝叶斯文档联合证据，也不是积分后的精确困惑度。所有模型定义一致；评估中不更改训练模型。词汇表外词不评分并单独记录 OOV 率。

## 6. 数值正确性与诊断

测试直接枚举极小语料的 Z 并对 π 做二维积分，用该真后验检验桌数 Gibbs；CRT 期望与概率归一化另行测试。合成数据由指定生成模型产生；先用 Hungarian 匹配 φ 的主题标签，再衡量 π 恢复误差，避免标签置换错误。

记录折叠 log p(Z,W,π)（含归一化常数）、每轮作者熵与最大归一化误差；每条链计算作者熵/对数联合轨迹 ESS，多链计算标签不变作者熵的 split R-hat。这些仅检查有限投影，不能证明高维主题链充分混合。多种子同时反映初始化及 Monte Carlo 变异，不能当作跨数据集不确定性。

## 7. 与既有工作关系

本项目使用有限层次 Dirichlet 主题模型及已有 CRT 数据扩充思想，不声称提出全新模型族。Author-Topic Model 直接由作者分布生成词主题；本模型额外保留文档 θ，允许同作者不同文档变化。λ 很大时接近单作者 Author-Topic 机制。固定 K 的作者局部层次结构也不同于全局无限主题的 HDP。

- Blei, Ng, Jordan (2003), [Latent Dirichlet Allocation](https://www.jmlr.org/papers/v3/blei03a.html).
- Rosen-Zvi et al. (2004), [The Author-Topic Model for Authors and Documents](https://cocosci.princeton.edu/tom/papers/author1.pdf).
- Zhou & Carin (2012), [Augment-and-Conquer Negative Binomial Processes](https://papers.nips.cc/paper_files/paper/2012/hash/6a61d423d02a1c56250dc23ae7ff12f3-Abstract.html).
- Zhou et al. (2016), [Augmentable Gamma Belief Networks](https://jmlr.csail.mit.edu/papers/volume17/15-633/15-633.pdf), CRT 定义及 Bernoulli 分解。这里通过上面的升阶乘恒等式独立给出适用于本模型的推导。

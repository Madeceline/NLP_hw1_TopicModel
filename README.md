# AP-LDA：基于作者主题偏好的主题建模

本项目从零实现 **CRT 辅助变量 + 折叠 Gibbs 采样**的 Author-Prior LDA，保留文档先验 `θ_d | π_a ~ Dir(λπ_a)`，从文本联合估计潜在作者偏好。包括 sklearn LDA、相同 Gibbs 算法的对称先验 LDA、保留作者频次的标签置换对照，真实语料与合成恢复严格分开。

**建议先打开 [演示.ipynb](演示.ipynb)**：16 步中文讲解，依次解释公式、代码、实际输出与结论限制。Notebook 默认复用已有数据和模型，保留合成恢复、局部推断、验证评价及测试的实际运行输出；也可打开重训开关。

## 安装和使用

Python 3.10–3.12；本次使用 Python 3.12.14。核心依赖与本次环境版本固定在 requirements.txt。

```bash
python -m pip install -r requirements-notebook.txt
jupyter lab 演示.ipynb
```

只需要终端研究流程时，安装 `requirements.txt` 即可，无需 Jupyter。首次运行 Numba 会编译采样核。

Notebook 运行依赖以其所选的内核环境为准。如果测试步骤提示 `No module named pytest`，在 Notebook 代码单元执行 `%pip install pytest==7.4.4`，然后重新运行测试步骤即可；不用重新安装已有的科学计算库。

```bash
# 从配置准备数据、运行全部实验、生成图和中文报告
python -m scripts.run_experiments --config configs/default.yaml

# 运行数学/数值/接口测试
python -m pytest -q

# 在已处理的词袋上训练一个 AP-LDA 并导出作者/文档主题
python -m scripts.train --topics 10 --lambda-author 10 --output results/custom_training

# 对一个已训练模型做统一文档完成评价
python -m scripts.evaluate --checkpoint results/custom_training/model.npz

```

命令均在项目根目录运行。实验脚本可自动复用相同配置的 checkpoint 与指标；`--no-resume` 重训。改变 YAML 时用 `--output results_new` 指定新的**项目产物根目录**，该目录会包含自己的 data/results/reports；不允许把不同配置混写进同一 results 目录。脚本只读取测试词来评分，不用测试集选择配置。

## API

```python
from src.models import AuthorPriorLDA

# documents: 稀疏整数词频矩阵，或 list[list[int]] 词 ID 序列。
# author_ids: 每篇文档一个作者标识；新文档缺失作者可使用 None。
model = AuthorPriorLDA(
    n_topics=10, gamma=0.5, beta=0.01,
    lambda_author=10.0, random_state=42,
)
model.fit(documents, author_ids)
theta = model.transform(new_documents, new_author_ids)
pi = model.get_author_topic_distribution()
phi = model.get_topic_word_distribution()
model.save("results/model.npz")
restored = AuthorPriorLDA.load("results/model.npz")
```

作者分布行顺序见 `model.author_ids_`，训练文档主题见 `model.document_topics_`。新文档必须使用训练时相同的词 ID/矩阵列。原始文本先分词并构造训练词表；新词在词表构建层处理，不把超出范围的 ID 交给模型。同一个随机种子、同一批输入的结果可重复；不同批次组合可能改变局部 Monte Carlo 抽样序列。

未知作者采用每篇文档内的层次先验 CRT 推断；已知作者采用训练 π 后验均值；空观察文档返回其先验期望。预测期间不更新全局 φ 或已知作者 π。保存格式是 NPZ+JSON，无 pickle 代码执行。Dirichlet 参数严格为正，极端超参数导致浮点采样下溢时明确报错。

自己的英文语料可使用 JSONL：每行 `{"id":"doc001","author":"Author A","text":"..."}`。

```bash
python -m scripts.train --jsonl my_documents.jsonl --topics 10 --output results/my_model
```

当前预处理器面向英文，中文语料需要适配分词器后再编码。模型核心不依赖语言。

## 实验与评价

真实语料：[UCI Reuter_50_50](https://archive.ics.uci.edu/dataset/217/reuter+50+50)，Liu (2006)，DOI [10.24432/C5DS42](https://doi.org/10.24432/C5DS42)。UCI 页面标示 CC BY 4.0；原始包共 50 记者、5000 篇新闻。下载 URL、SHA256、来源引用在 `data/raw/provenance.json`。

默认选择 12 作者，保留官方 train/test 边界，构造 384/144/240 篇训练/验证/测试，训练词表 1800 词，每篇上限 220 个词位。λ 网格 `[1,5,10,20,50,100]`，K 网格 `[5,10,15,20,30]`，三种子 `[11,29,47]`；顺序按验证分数选择 λ，再选择 K。作者文档数实验用 `[4,12,32]`。所有真实配置共 144 次训练，另有 6 次合成验证和 3 条长链诊断。

所有模型使用同一观察词/预测词划分、同一局部推断器和同一 plug-in 单词边缘预测分数；另计算训练文档窗口 NPMI、多样性。测试分别观察 10%/50% 的词位。均值±标准差是三种子的样本标准差，不代表跨语料区间；报告另提供探索性作者簇 bootstrap。

**数值限制**：主实验每链 240 轮，真实语料不同链的混合诊断偏离理想状态，不能把有限预算的主题估计视作已充分收敛的精确后验。额外 2400 轮链只用于验证诊断，不更改测试设置。实际数值和限制见 [最终报告](reports/final_report.md) 与 [诊断文件](results/metrics/mcmc_diagnostics.json)。不预设 AP-LDA 全面优于 LDA。

## 产物索引

| 入口 | 内容 |
| --- | --- |
| [逐步 Notebook](演示.ipynb) | 中文教学、可执行代码与真实输出 |
| [数学推导](reports/mathematical_model.md) | 联合分布、CRT 推导、条件更新、理论性质 |
| [研究报告](reports/final_report.md) | 实际实验表格、图、解释、优势和限制 |
| `results/metrics/validation_runs.csv` | 全部超参数/稀疏性配置的三种子原始指标 |
| `results/metrics/test_scores.csv` | 固定和验证选择协议下的实际测试指标 |
| `results/metrics/test_document_scores.csv` | 可审计的逐文档预测分数与词数 |
| `results/distributions/` | 全作者 π、全文档 θ、主题词 φ 和关键词 |
| `results/checkpoints/` | 全部真实配置模型（NPZ+JSON） |
| `results/figures/` | PNG/SVG 图；DOT 源图和字体/后端说明 |
| `results/config_snapshot.yaml` | 实际执行配置快照 |
| `results/logs/` | 运行、测试与 Notebook 执行记录 |
| `data/processed/` | 独立划分、词表、文档作者映射、数据审计 |

图使用本地中文字体。优先 Graphviz；无系统 `dot` 时回退 Matplotlib，仍生成 PNG/SVG 和 DOT 源码，不需要图像生成服务。

## 技术取舍

CRT 采样利用已有数据扩充思想，非原始空间的简单共轭，也不是对普通 LDA 包装作者输入；具体证明见数学文档。与 Rosen-Zvi 等 (2004) Author-Topic Model 的区别是保留了每篇文档 θ。与层次 Dirichlet/HDP 有联系，但固定有限 K，不声称全新模型族。

sklearn 1.5 的先验校验将 α 限制为 ≤1；为测试 λ/K>1 的数学合法浓度，局部通过其公开配置关闭元数据校验，仍由本模型检查参数正值，未改写其变分更新。LDA-Gibbs 提供相同采样算法对照。各模型训练轮次单位不同，因此记录实际时间而不声称预算严格相等。

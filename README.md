# 可认证模糊粗糙属性约简：研究工作区

当前主稿研究：查询与参考对象共同发生区间扰动时，留一模糊粗糙分类间隔的投影混合松弛、渐近紧的误差界与决策保持条件。最新审阅稿在 `paper-round4/main.tex` / `main.pdf`；从 `research/round4/核心增量与模糊粗糙价值.md` 开始。

第四轮重要修正：新取得Haines2011全文，确认其证明可直接适配本稿L1困难性及非零gap，因此撤回“独立核心困难性创新”的定位。当前只保留投影缩域后的LP等价、显式误差与紧性这一狭义候选组合，尚不能断言达到FSS新颖性门槛。1536个比较表明平均逐敌误差界降低约75.1%，充分精确条件覆盖170→218个对象；正认证仍73个，无新增分类优势。历史稿与ZIP未覆盖。

前两轮研究的是同一扰动实现下的依赖度保持与属性约简，现保留为历史快照。这些模型和第三轮的签名分类间隔不同，实验不能混并。

这是包含真实实验和英文稿件的研究工作区，尚未投稿。第三轮已完成主要证明的独立自动复核、384个合成认证问题、140个真实小表问题、15个数据集的首折探索，以及303项测试；没有证明预测性能优势。作者信息按用户指定的 Electronics PDF 核对，最终科学审阅与本研究的投稿声明仍由真实作者确认。2026-09-20完成的第二轮逐类相对保持与子集归一化探索保留在原目录及ZIP中。

## 从这里开始

| 内容 | 位置 |
|---|---|
| 当前核心定理增量与价值 | `research/round4/核心增量与模糊粗糙价值.md` |
| 最新第四轮英文审阅稿 | `paper-round4/main.pdf`，源文件 `paper-round4/main.tex` |
| 第四轮逐篇定理查新与数学复核 | `research/round4/theorem-delta-audit.md`、`manuscript-math-review.md` |
| 第四轮投影LP核验 | `research/round4/projected-mixture-results.md`、`results/round4/projected-mixture-audit.json` |
| 第三轮历史研究与投稿清单 | `research/round3/研究进展与投稿清单.md` |
| 第三轮英文历史稿 | `paper-round3/main.pdf`，源文件 `paper-round3/main.tex` |
| 第三轮复现说明 | `submission/round3/reproducibility.md` |
| 第三轮理论与文献审计 | `research/round3/final-theory-audit.md`、`theorem-novelty-check.md` |
| 第二轮研究与决策 | `research/round2/研究进展与决策.md` |
| 第一轮结论与投稿准备 | `研究结果与投稿准备.md` |
| 第二轮证明与独立审计 | `research/round2/relative-theory-independent.md`、`normalized-relative-theory.md`、`normalized-numerical-audit.md` |
| 第二轮小表穷举审计 | `research/round2/normalized-subset-audit.md` |
| 非单调搜索障碍与整表候选试验 | `research/round2/nonmonotone-search-barrier.md`、`normalized-candidate-pilot-results.md` |
| 第一轮英文历史稿 | `paper/main.pdf`，源文件 `paper/main.tex` |
| 文献下载目录与状态 | `research/literature/library.json`、`research/literature/pdf/` |
| 新补八篇正式全文核查 | `research/literature/fulltext-update.md`（含分篇笔记、MAFRFS正式版差异和附件缺口） |
| 精读笔记与新颖性核对 | `research/literature/reading-notes.md`、`novelty-matrix.md` |
| 证明与适用条件 | `research/theory.md`、`paper/theory.tex` |
| 实验方案 | `research/experimental-protocol.md` |
| 原始结果 | `results/primary/`、`results/sensitivity/` |
| 全部数据集结果表 | `results/dataset-means.csv` |
| 数据来源与许可 | `data/manifest.json`、`research/data-and-baselines.md` |
| 近期框架复现边界 | `research/mafrfs-reproduction.md` |
| 独立审查 | `research/independent-review.md` |

## 环境与复现

Python 3.11。工作区 `.venv` 已安装运行依赖；精确版本保存在 `requirements-lock.txt`。项目主实现位于 `src/certified_frfs/`，脚本自动加入该路径，不依赖全局安装本项目。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe data/download_datasets.py
.\.venv\Scripts\python.exe scripts/run_experiments.py --workers 3
.\.venv\Scripts\python.exe scripts/run_sensitivity.py --workers 2
.\.venv\Scripts\python.exe scripts/summarize_results.py
.\.venv\Scripts\python.exe scripts/build_paper_results.py
.\.venv\Scripts\python.exe scripts/verify_reproduction.py
```

完整实验已经运行，重跑时会核查缓存与参数、代码指纹；不匹配则停止，避免错误复用旧结果。进行新的参数实验请用新 `--output` 目录。敏感性脚本默认目录固定，代码或协议有变更时先归档旧结果。不要用新的配置覆盖原实验。

理论校验：

```powershell
.\.venv\Scripts\python.exe scripts/run_prototypes.py
.\.venv\Scripts\python.exe scripts/validate_theory.py
.\.venv\Scripts\python.exe scripts/explore_exact_stability.py --source results/primary
```

`run_prototypes.py` 的 MILP 结果是数值容差意义下的全局界；随机攻击只提供下界。更多探索脚本和对应输出在 `scripts/` 与 `results/`。默认主实验不启用精确稳定性探索。

第二轮运行入口：`scripts/run_relative_prototypes.py`、`run_relative_pilot.py`、`run_normalized_prototypes.py`、`run_normalized_pilot.py`、`audit_normalized_subsets.py`。结果单独保存在`results/round2/`；重跑前归档同名输出。218项完整测试通过，证书仍为浮点实现，不能称形式化验证。两份论文附件无法取得，按用户要求继续独立研究，不再作为前置条件。

进一步审计：730个小表真子集的2528次类级MILP核查得到98可行、632不可行；加强稳定性检查仅额外认证2个。3个可行子集存在逐列删除路径障碍，WDBC小表需一次删两列才能达到其1属性解。随后在15个整训练表直接检查431个单属性/MI前缀等候选，没有改善旧贪心输出，见`scripts/run_normalized_candidate_pilot.py`。整合核对记录为`results/round2/integration-audit.json`。

第二轮单独复现包：`output/certified-frfs-round2-exploration.zip`，生成脚本`scripts/package_round2.py`；第一轮v1包未覆盖。

## 重要解释

- 第三轮保证对象是有限表上的留一分类间隔；前两轮保证的是依赖度。三轮均固定标签，均不提供总体泛化风险界。
- 不把“未找到攻击”当作证明。
- 当前浮点实现使用安全余量，不是严格区间算术认证。
- 约简输出是可认证的非空可行子集，不声称全局最小或包含极小。
- `maf_framework` 是明确指定加性依赖度的框架实例，不是作者原版 FD+ 复现；RFRR等核心正文已补齐，两份附件缺失仍限制原版直接对照。
- 原始 UCI 数据保留出处及 CC BY 4.0 许可证据。公开发布时仍需逐数据集归属说明；此工作区未公开上传。
- 文献PDF包含合法开放版本与用户提供的正式全文，不应将版权不同的全文与代码一起任意再分发。

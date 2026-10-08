# 多阶段推荐系统与历史行为实验

基于原作者 ChendiLiu https://github.com/1-dr-eam/RecommenderSystem.git的推荐系统项目，保留多路召回、粗排、精排、重排的完整链路，扩展时间切分、离线训练、方法切换、实验记录和用户历史行为特征。

本项目用于推荐算法学习和工程实验。数据以模拟用户和交互为主，离线指标不代表真实业务收益。

## 来源与贡献

原作者提供了多阶段推荐系统的基础工程。原始说明保存在 [README_UPSTREAM.md](README_UPSTREAM.md)，版权和许可见 [LICENSE](LICENSE)。原项目参考了[王树森的推荐系统课程](https://www.bilibili.com/video/BV1HZ421U77y/)和 [OpenAI CLIP](https://github.com/openai/CLIP)。

| 部分 | 原作者成果 | 本实验副本的改动 |
|---|---|---|
| 召回 | UserCF、ItemCF、Swing、DCN 双塔、类目/关键词、内容聚类、LightGCN；FAISS 检索 | 修复聚类召回对象与 ID 混用；新增历史兴趣召回 |
| 粗排 | 三塔模型，缓存物品塔输出 | 融合历史兴趣分数；支持方法切换 |
| 精排 | 多目标 DCN；保留其他模型定义供扩展 | 融合历史兴趣分数；支持方法切换 |
| 重排 | CLIP 图文向量与 MMR 多样性重排 | 本次历史特征实验沿用原算法 |
| 训练与数据 | 模型训练函数、CSV 数据和预置权重 | 时间切分、分阶段训练入口、配置记录、权重副本、最终重训 |
| 评估 | 原项目侧重链路与服务实现 | 验证集选参、测试评估、指标比较、实验编排 |
| 工程 | FastAPI、数据库适配、增量微调接口 | 保留服务代码；本地实验入口另行扩展，未同步改造数据库服务的实验流程 |

当前代码还包含数据加载 worker、批量传输和部分内容特征/相似度计算的性能优化；实际运行参数以实验配置和代码为准。

## 架构与目录

```text
用户画像 + 物品内容 + 历史交互
                |
             多路召回
                |
            三塔模型粗排
                |
          多目标 DCN 精排
                |
           MMR 多样性重排
                |
          有序推荐物品 ID
```

历史序列适配器影响召回、粗排和精排，不单独修改重排公式。精排分数改变后，最终 MMR 输出也可能改变。

| 路径 | 用途 |
|---|---|
| [main.py](main.py) | CSV 数据准备、模型加载、端到端推荐 |
| [recall.py](recall.py) | 多路召回；双塔与图模型分别位于 `twin_towers_model.py`、`LightGCN.py` |
| [rough_ranking.py](rough_ranking.py)、[fine_ranking.py](fine_ranking.py) | 粗排、精排训练与推理 |
| [rearrangement.py](rearrangement.py)、[entities.py](entities.py) | MMR、物品内容特征和用户画像 |
| [feature_processor.py](feature_processor.py)、[dataset.py](dataset.py) | 词表、缩放和训练样本 |
| [history_features.py](history_features.py) | 历史兴趣编码与 `HistoryConfig` |
| [methods/registry.py](methods/registry.py) | 各阶段 baseline/improved 适配器 |
| [data_split.py](data_split.py)、`data/splits/` | 时间切分与元数据 |
| [training/runner.py](training/runner.py) | 按阶段训练、记录配置、导出权重 |
| [training/performance.py](training/performance.py) | DataLoader 设置与训练耗时日志 |
| [evaluation/evaluate.py](evaluation/evaluate.py)、[evaluation/metrics.py](evaluation/metrics.py) | 完整链路评估与 Top-K 指标 |
| [evaluation/search_history.py](evaluation/search_history.py)、[evaluation/compare.py](evaluation/compare.py) | 验证集搜索与结果比较 |
| [training/pipeline.py](training/pipeline.py) | 串联训练、选参、重训和最终评估 |
| `experiments/`、`model_weights/` | 实验产物、原项目附带权重 |
| `models/`、`interface/`、`sql/` | 扩展模型定义、原服务与数据库代码、DDL |

## 数据与实验规则

数据位于 `data/users_new.csv`、`data/items_new.csv`、`data/interactions_new.csv`。部分书籍信息真实，用户和交互主要为模拟数据。

默认按每个用户的时间顺序留出最新一条作为测试、倒数第二条作为验证，其余作为训练。记录不足时，相应训练或验证集合可能为空。这是按用户留出的时间切分，不是所有用户共享同一全局时间截点。

| 文件 | 用途 | 当前记录数 |
|---|---|---:|
| `interactions_train.csv` | 初始训练、验证请求的历史 | 74,946 |
| `interactions_validation.csv` | 选参和开发检查 | 9,989 |
| `interactions_train_validation.csv` | 选参后的最终重训与历史 | 84,935 |
| `interactions_test.csv` | 固定方案的最终评估 | 9,999 |

数量来自当前 [split_metadata.json](data/splits/split_metadata.json)。数据变化后以重新生成的元数据为准。

建议流程为“train 训练 → validation 选参 → train+validation 重训 → 固定两个方案 → test 对比”。切分脚本会读取原始全量数据生成文件；训练、搜索和常规推荐不读取测试标签。选参后不要根据测试指标继续调参。

## 环境与首次运行

以下命令从 `experiment_project/` 执行，示例使用 Linux/Bash。Python 3.10 及以上支持当前类型注解。

[dependency/environment.yml](dependency/environment.yml) 是原环境的参考清单，包含 Windows 专用构建和非必要依赖，Linux 不宜直接照搬。为机器配置匹配的 PyTorch/torchvision 和 FAISS，再安装 pandas、NumPy、scikit-learn、Pillow、requests 等依赖。`clip` 必须是提供 `clip.load` 的 OpenAI CLIP，不能仅凭同名 PyPI 包判断安装正确。首次启动可能下载 CLIP 权重和物品图片，需联网或预置缓存。

```bash
python -c "import torch, pandas, sklearn, faiss, clip; print(torch.__version__, torch.cuda.is_available()); print(hasattr(clip, 'load'))"
```

已训练时不要随意重新切分或修改用户/物品数据。当前模型加载会重新构建词表，数据变化可能使 embedding 尺寸或 ID 对应关系失配。

## 运行顺序

### 1. 生成切分并训练

仅在首次准备数据或明确要重建实验时生成切分：

```bash
python data_split.py
python -m training.runner --stage all --interaction-split train --experiment-name baseline_train
```

`--stage` 支持 `recall`、`rough_ranking`、`fine_ranking`、`all`。单阶段产物不包含全链路所需的其他权重。训练复用原函数，先写入 `model_weights/`，再复制到实验目录；因此训练也会更新副本中的默认权重目录，比较时应使用实验导出的权重。

把输出的实际路径填入变量：

```bash
WEIGHTS="experiments/实际训练目录/weights"
SYSTEM_SPLIT=train
```

### 2. 检查推荐与验证链路

```bash
python main.py --weights-dir "$WEIGHTS" --method-profile baseline
python main.py --weights-dir "$WEIGHTS" --method-profile history --history-mode hybrid
python -m evaluation.evaluate --weights-dir "$WEIGHTS" --split validation --system-interaction-split train --method-profile baseline --max-users 20 --experiment-name validation_smoke
```

`main.py` 默认用 train 构建模型和历史，启动后不会自动重训。原项目预置权重的数据来源未由实验配置确认，不能保证与当前切分匹配；优先使用上述重新训练的权重。

`--max-users` 选择切分文件中先出现的 N 个用户，不是随机抽样。小规模结果用于检查链路，不能直接代表全量效果。失败请求会记录在 `evaluation.json` 中并按空推荐计分。

### 3. 可选：验证集搜索与最终重训

先用少量用户、单组参数检查搜索流程：

```bash
python -m evaluation.search_history --weights-dir "$WEIGHTS" --max-users 20 --modes hybrid --lengths 20 --alphas 2.0 --hard-topks 10 --temperatures 0.1 --ranking-weights 0.15 --experiment-name search_smoke
```

正式选参可扩大范围并移除用户限制。默认搜索 32 组配置，每组运行完整推荐链路；1 万用户约需 32 万次请求。当前无断点续跑，结果在全部搜索完成后保存。

选择最佳配置后：

```bash
python -m training.runner --stage all --interaction-split train_validation --experiment-name final_train_validation
```

此时将 `WEIGHTS` 更新为最终重训的权重目录，并设 `SYSTEM_SPLIT=train_validation`。若时间有限，可以跳过选参与重训，固定默认 HistoryConfig，继续使用 train 权重和 train 历史；应明确报告“预设参数实验”，而非“验证集最优参数”。

### 4. 固定方案，评估 baseline 与 history

两次使用同一权重、历史切分、K 和用户范围，隔离历史适配器的作用：

```bash
python -m evaluation.evaluate --weights-dir "$WEIGHTS" --split test --system-interaction-split "$SYSTEM_SPLIT" --method-profile baseline --k 10 --experiment-name test_baseline
python -m evaluation.evaluate --weights-dir "$WEIGHTS" --split test --system-interaction-split "$SYSTEM_SPLIT" --method-profile history --history-mode hybrid --history-L 20 --history-alpha 2.0 --k 10 --experiment-name test_history_hybrid
```

如已选参，history 命令应使用 `--history-config experiments/实际搜索目录/best_history_config.json`，替换手动 mode/L/alpha 参数，以载入全部选定参数。

```bash
python -m evaluation.compare --baseline experiments/实际baseline评估目录/evaluation.json --improved experiments/实际history评估目录/evaluation.json > experiments/test_comparison.json
```

`compare` 输出绝对差值，不会自动核对权重、用户范围、失败率或统计显著性，比较前需检查原始文件。

### 实验产物在哪里

| 产物 | 路径 |
|---|---|
| 训练配置、实际设置和权重路径 | `experiments/<UTC时间>_<名称>/config.json` |
| 导出权重 | 同目录 `weights/` |
| 搜索全部配置与最佳配置 | 搜索目录 `search_results.json`、`best_history_config.json` |
| 单次指标、失败示例和方法配置 | 评估目录 `evaluation.json` |
| 两方案差值 | 上述命令输出的 `experiments/test_comparison.json` |
| 自动编排总记录 | pipeline 目录 `experiment_manifest.json` |

`python -m training.pipeline --experiment-name history_protocol --max-users 200` 可自动执行 train 训练、baseline 验证、history 选参、train_validation 重训和 history 测试。**当前 pipeline 只在最终测试阶段评估 history，没有自动生成最终 baseline 测试结果或双方案比较**；需要比较时仍按第 4 步运行。CLI 配置记录不是通用训练超参数注入器，JSON 中的 seed 等字段不能视为已自动应用；训练设置以实际训练函数及日志为准。

## 历史特征的实现

| 模式 | 当前实现 |
|---|---|
| `din` | 候选内容向量与完整历史点积，经温度 softmax 后加权 |
| `sim_soft` | 当前与 din 使用相同的相似度 softmax 计算 |
| `sim_hard` | 按相似度选 hard_topk 条历史，再做注意力加权 |
| `hybrid` | 长度 <= L 走 din；L < 长度 <= alpha*L 走 soft；更长走 hard |

默认 `HistoryConfig`：mode=hybrid、L=20、alpha=2.0、hard_topk=10、temperature=0.1、recall_topk=50、ranking_weight=0.15。L 是分支阈值，不是当前编码器的强制截断长度。

召回在原候选集上增加历史内容相近的候选；粗排和精排融合历史分数。当前编码器没有独立可训练的 DIN 注意力网络或完整 SIM 两阶段训练过程，也没有把序列直接拼入原神经网络进行端到端训练。应称为“DIN/SIM 思路的历史兴趣适配器”，而非完整论文复现。仅 `mode=hybrid` 会按长度切换分支。

## 已完成的离线实验

2026-10-08 的远程评估使用同一套 train 权重和 train 历史，测试 9,999 个用户，K=10。history 使用上述默认 hybrid 配置，未完成验证集超参数搜索，也未执行 train_validation 最终重训。

| 指标 | baseline | history hybrid | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| HitRate@10 | 0.5301% | 0.7101% | +0.1800 个百分点 | +33.96% |
| Recall@10 | 0.5301% | 0.7101% | +0.1800 个百分点 | +33.96% |
| NDCG@10 | 0.002205 | 0.003688 | +0.001483 | +67.25% |

原始文件：[baseline](experiments/20261008T035453Z_test_baseline/evaluation.json)、[history hybrid](experiments/20261008T052313Z_test_history_hybrid/evaluation.json)。

两次均有 **44 个失败用户**，示例错误为 `TypeError: unsupported operand type(s) for |: 'list' and 'list'`。指标分母仍为 9,999，失败请求按未命中计分；原始文件只保存少量失败示例，不能确认全部失败用户集合完全一致。需要处理该链路问题后再开展后续验证，不能将本结果描述为全用户无异常运行。

每用户仅一个测试目标，HitRate 和 Recall 因而相同；53 个命中变为 71 个，净增加 18 个。尚未做显著性检验或线上 A/B 测试，结果只说明这组配置在当前模拟数据上提供了正向离线信号。

## 已知限制与后续方向

- 模型词表和 scaler 未作为独立工件持久化，加载依赖同一切分和数据快照；`size mismatch` 时核对训练配置及 `--system-interaction-split`，不要强行截断 embedding。
- 按用户留出不等同于严格全局时间回放；用户/物品元数据与统计特征也尚未证明按请求时刻重建。
- 当前评估汇总每用户目标集，并使用该用户首条留出记录的场景；增加多条留出记录并不会自动变成逐事件回放。
- 训练入口复用原训练循环，尚无统一逐 epoch 验证、early stopping 或最佳 checkpoint 选择。
- history 搜索包含对部分模式无效的 L/alpha 重复组合，且未提供中途保存或实时试验进度。
- 下一步可补充逐阶段候选命中率、逐用户结果、分支覆盖率、配对统计检验和请求延迟，再定位效果与性能瓶颈。

## 服务与许可

`interface/` 保留原作者的 FastAPI、数据库和微调接口，`local_api.py` 提供本地接口相关代码。服务部署需额外配置数据库和运行环境；离线评估无需启动 API。本次历史改进实验以 CSV 主链路为准，未验证所有服务入口同样支持改进配置。

代码使用 [MIT License](LICENSE)，保留原版权声明 `Copyright (c) 2026 ChendiLiu`。原 README 声明预置模型权重也使用 MIT License。本副本的实验框架和历史特征扩展建立在原作者成果之上，不将既有模型与服务实现归为新增贡献。

完整分步说明另见 [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)；使用时以当前代码、实验配置和本 README 的已知限制为准。

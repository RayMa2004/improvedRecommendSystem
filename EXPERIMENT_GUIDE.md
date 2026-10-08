# 实验版项目使用说明

本目录是原项目的隔离副本。根目录原项目保持不变，后续改动只在这里进行。

## 1. 生成时间切分

在本目录根目录运行：

```bash
python data_split.py
```

划分规则是按用户和时间排序：

- 较早交互：`data/splits/interactions_train.csv`
- 每个用户倒数第二条交互：`data/splits/interactions_validation.csv`
- 每个用户最后一条交互：`data/splits/interactions_test.csv`
- 训练集和验证集合并：`data/splits/interactions_train_validation.csv`，只用于最终重训

推荐系统正常启动时只读取训练集。测试集只在 `evaluation` 模块中读取，用于模拟未来请求。

## 2. 运行当前 baseline

```bash
python main.py
```

当前入口会优先读取训练切分，并把训练交互填入用户最近历史队列。仓库附带的 `model_weights/` 是原项目的历史权重；要做严格的无测试泄漏实验，应先用下面的离线训练板块基于训练集重新生成一套权重。

## 3. 离线训练

训练板块复用原有模型训练 method：

```bash
python -m training.runner --stage recall --experiment-name recall_baseline
python -m training.runner --stage rough_ranking --experiment-name rough_baseline
python -m training.runner --stage fine_ranking --experiment-name fine_baseline
python -m training.runner --stage all --experiment-name full_baseline
```

严格实验建议使用：

```bash
python -m training.runner --stage all --experiment-name full_baseline --config experiments/configs/baseline.json
```

每次训练会在 `experiments/<时间>_<名称>/` 下保存：

- `config.json`：实验名称、时间、数据切分、method 选择和配置；
- `weights/`：本次训练导出的权重副本。

训练方法只使用指定的训练切分。严格实验中先指定 `train`，选参后再指定
`train_validation`，从而避免测试集参与训练或选参。

## 4. 评估验证集或测试集

先用少量用户做快速检查：

```bash
python -m evaluation.evaluate --split validation --max-users 20 --experiment-name validation_smoke
```

完整测试集评估：

```bash
python -m evaluation.evaluate --split test --experiment-name test_baseline
```

输出包括 `HitRate@K`、`Recall@K`、`NDCG@K` 和失败用户数量，并保存为 `evaluation.json`。

`--max-users` 会同时限制推荐用户和指标统计用户，避免未生成推荐的用户被当成 0 分。

两次评估完成后，可以直接比较指标变化：

```bash
python -m evaluation.compare \
  --baseline experiments/<baseline时间>_test_baseline/evaluation.json \
  --improved experiments/<improved时间>_test_improved/evaluation.json
```

## 5. 切换 baseline / improved method

编辑 [`methods/registry.py`](methods/registry.py) 中的：

```python
ACTIVE_METHODS = {
    "recall": "baseline",
    "rough_ranking": "baseline",
    "fine_ranking": "baseline",
    "rearrangement": "baseline",
}
```

当前每个阶段的 `improved` 是可替换入口。历史序列 improvement 已实现召回、粗排和精排适配器；重排继续使用 baseline。

## 6. 历史序列 improvement

历史序列实现位于 [`history_features.py`](history_features.py)，使用物品已有的 CLIP 内容向量作为序列元素：

- `din`：对完整历史序列做候选物品条件注意力；
- `sim_soft`：对完整历史序列做 soft similarity search；
- `sim_hard`：只保留相似度最高的 `hard_topk` 条历史行为；
- `hybrid`：长度不超过 `L` 使用 DIN，超过 `L` 使用 SIM(soft)，超过 `alpha * L` 使用 SIM(hard)。

默认参数是 `L=20`、`alpha=2.0`。修改 `HistoryConfig` 即可进行参数实验。

历史序列 improvement 会影响召回、粗排和精排，重排仍然只使用原有 CLIP 物品相似度。要启用它，可以把 `methods/registry.py` 中的 `ACTIVE_METHODS` 改为：

```python
from methods.registry import HISTORY_IMPROVEMENT_METHODS
ACTIVE_METHODS = dict(HISTORY_IMPROVEMENT_METHODS)
```

也可以不改文件，直接在评估时切换：

```bash
python -m evaluation.evaluate --split test --method-profile baseline --experiment-name test_baseline
python -m evaluation.evaluate --split test --method-profile history --history-mode hybrid --history-L 20 --history-alpha 2.0 --experiment-name test_history
```

## 7. 按正确实验流程自动选参并最终测试

推荐使用以下顺序：训练集训练，验证集选择 `HistoryConfig`，训练集加验证集重训，
最后只读取一次测试集。

先训练 baseline 权重：

```bash
python -m training.runner --stage all --interaction-split train --experiment-name protocol_train
```

用验证集搜索历史序列参数。搜索过程只读取 `interactions_validation.csv`，不能把
`--split` 改成测试集：

```bash
python -m evaluation.search_history \
  --weights-dir experiments/<训练实验目录>/weights \
  --split validation --metric ndcg --k 10 --max-users 200
```

搜索结果目录中会保存 `search_results.json` 和 `best_history_config.json`。找到最佳配置后，
用训练集加验证集重新训练各阶段模型：

```bash
python -m training.runner --stage all \
  --interaction-split train_validation \
  --experiment-name protocol_train_validation
```

最后固定 `best_history_config.json` 中的配置，只做一次测试集评估：

```bash
python -m evaluation.evaluate \
  --weights-dir experiments/<最终重训目录>/weights \
  --split test --method-profile history \
  --system-interaction-split train_validation \
  --history-config experiments/<搜索实验目录>/best_history_config.json \
  --experiment-name protocol_test
```

也可以由一个命令自动编排上述流程，并在最终实验目录保存
`experiment_manifest.json`：

```bash
python -m training.pipeline --experiment-name history_protocol --max-users 200
```

该编排脚本会在测试步骤之前完成所有训练和验证选参；测试集只在最后的评估步骤读取。

## 8. 粗排、精排的并行数据加载

排序训练默认使用 `batch_size=1024`、`num_workers=8`。`ThreeTowerDataset`
输出 CPU 张量，worker 负责数据准备；主进程通过 pinned memory 和非阻塞传输将完整
batch 搬到 GPU。worker 使用 `spawn` 启动，跨 epoch 保持存活并预取数据。

远程机器需要同步以下代码文件：

- `rough_ranking.py`
- `fine_ranking.py`
- `dataset.py`
- `feature_processor.py`
- `rearrangement.py`
- `training/performance.py`（新增，必须上传）
- `training/runner.py`

保留远程已有的 `model_weights/` 和 `experiments/`，仅更新代码文件。运行示例：

```bash
cd /mnt/experiment_project
export ROUGH_RANKING_NUM_WORKERS=8
export ROUGH_RANKING_BATCH_SIZE=1024
export FINE_RANKING_NUM_WORKERS=8
export FINE_RANKING_BATCH_SIZE=1024

python -m training.runner --stage rough_ranking --interaction-split train --experiment-name rough_parallel
python -m training.runner --stage fine_ranking --interaction-split train --experiment-name fine_parallel
```

worker 数量可以改为 `4`、`8` 或 `16`，需结合机器可用 CPU 核数和耗时测量选择。
显存不足时将 batch size 改为 `512` 或 `256`。设置 worker 数量为 `0` 可以排查
多进程错误。微调入口也使用相同机制，独立参数前缀为 `ROUGH_RANKING_FINETUNE`
和 `FINE_RANKING_FINETUNE`。

终端日志包括加载参数、输入设备、GPU 型号、显存占用、epoch 耗时和 `data_wait_s`。
第一个 epoch 的等待时间包含 worker 启动开销，后续 epoch 更适合衡量吞吐量。
`experiments/<训练目录>/config.json` 的 `default_training_settings` 会记录实际使用的
batch size、worker 数量、pinned memory 和预取设置；baseline 和 improvement
比较时应保持这些配置一致。

分别训练的实验目录仅导出对应阶段的权重；完整评估需要四个模型权重齐全的统一目录。
项目 `model_weights/` 会在各阶段训练时更新，也可以将各实验导出的权重复制到同一个目录。

## 9. 重排的 GPU 离线计算

MMR 重排没有梯度训练，其离线环节是建立物品余弦相似度矩阵，在 `main.py` 或评估系统
初始化时自动执行。默认 GPU 分块大小从 `512` 增加到 `2048`，也可调整为：

```bash
export MMR_SIMILARITY_BATCH_SIZE=4096
python main.py --weights-dir model_weights --method-profile baseline
```

相似度矩阵计算日志会显示实际设备、分块大小和总耗时。GPU 将矩阵计算并行执行，
分块结果保存在 CPU；在线 MMR 选择规则继续使用原有实现。

已新增 worker 和分块计算回归测试，远程环境可执行：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

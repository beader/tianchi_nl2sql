# 首届中文NL2SQL挑战赛

团队名: Model S

### 环境依赖

深度学习框架: tensorflow, keras

Docker 镜像:

|REPOSITORY|TAG|IMAGE ID|
|:---:|:---:|:---:|
|tensorflow/tensorflow|nightly-gpu-py3-jupyter|6e60684e9aa4|

CUDA 版本: 10.0.130

Python 依赖:

见 `./code/requirements.txt`

通过该 sh 命令安装依赖
```
pip install -r ./code/requirements.txt
```

### 训练

NL2SQL 场景需要预测的结果包含以下几个部分:

```
- sel
- agg
- conds
  - [(col_id, cond_op, cond_val), ...]
```

训练将分为独立的两个任务进行。

- task1 负责对 `sel`, `agg`, `cond_conn_op` 以及 `conds` 当中的 `col_id`, `cond_op` 目标进行训练。
- task2 负责对 `conds` 中的 `col_id`, `cond_op`, `cond_val` 目标进行训练。


在 `./code` 目录下，执行

```
python task1.py train --model_dir ../model/
```

```
python task2.py train --model_dir ../model/
```


### 推断

推断的时候，先执行 task1，再执行 task2

在 `./code` 目录下，执行

```
python task1.py infer --model_weights ../model/task1.12-0.852.h5 --output_file ../submit/task1_output.json
```

```
python task2.py infer --model_weights ../model/task2.h5 --output_file ../submit/submit.json
```

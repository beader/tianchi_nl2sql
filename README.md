# 首届中文NL2SQL挑战赛

团队名: Model S

### 环境依赖

Docker 镜像:

|REPOSITORY|TAG|IMAGE ID|
|:---:|:---:|:---:|
|tensorflow/tensorflow|nightly-gpu-py3-jupyter|6e60684e9aa4|

python 依赖:

见 `./code/requirements.txt`

### 训练

在 `./code` 目录下，执行

```
python task1.py train --model_dir ../model/
```

```
#todo: task2 training phase
```


### 推断

在 `./code` 目录下，执行

```
python task1.py infer --model ../model/task1.12-0.852.hd5 --output_file ../submit/task1_output.json
```
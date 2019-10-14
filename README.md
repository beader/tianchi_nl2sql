# 首届中文NL2SQL挑战赛

[竞赛链接](https://tianchi.aliyun.com/competition/entrance/231716/introduction)

>:warning: 由于可能存在的版权问题，请自行联系竞赛平台或主办方索要竞赛数据，谢谢!

## 成绩

本项目所采用的方案在复赛中的线上排名为第五，决赛成绩为第三。主分支下的代码以 jupyter notebook 的形式呈现，以学习交流为目的，对原始的代码经过一定的整理，并不会完全复现线上的结果，但效果不会差太多。

## 致谢

- 感谢追一科技的孙宁远对本次比赛做了细致的赛前辅导。
- 感谢追一科技研究员、[科学空间](https://kexue.fm/)博主苏剑林，分享了大量关于NLP方面的优质博文。本方案受到了[基于Bert的NL2SQL模型：一个简明的Baseline]这篇文章的启发。项目中使用的 RAdam 优化器的实现直接来自于苏剑林开源的 [keras_radam](https://github.com/bojone/keras_radam/blob/master/radam.py) 项目。
- 感谢 CyberZHG 大神的开源项目 [CyberZHG/keras-bert]，本次比赛中我们使用了 keras-bert 构建我们的模型。
- 感谢哈工大讯飞联合实验室的 [Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm) 项目，本次比赛中我们使用了他们`BERT-wwm, Chinese` 预训练模型参数。

## 背景

首届中文NL2SQL挑战赛，使用金融以及通用领域的表格数据作为数据源，提供在此基础上标注的自然语言与SQL语句的匹配对，希望选手可以利用数据训练出可以准确转换自然语言到SQL的模型。

模型的输入为一个 Question + Table，输出一个 SQL 结构，该 SQL 结构对应一条 SQL 语句。

![](./imgs/terminology.png)


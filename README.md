# StanceGen

StanceGen是一个用于立场驱动的多模态回复生成和评估的框架，集成了多种大型语言模型的能力。

## 项目概述

StanceGen旨在利用先进的语言模型和多模态模型，进行立场驱动的内容生成、分析和评估。本项目介绍了立场驱动的多模态回复生成任务，目标是根据社交媒体上包含文本和视觉信息的帖子，生成立场一致的评论。

主要特点：
- 支持多种模型，包括Qwen（通义千问）、LLaVA（多模态大语言模型）、GLM/G系列模型
- 提供StanceGen2024数据集，包含2024年美国总统选举中的推文-图像/视频对及立场标注的用户评论
- 提出SDMG（Stance-Driven Multimodal Generation）框架，集成多模态特征的加权融合和立场引导
- 支持多种评估方法，包括相似度（COS）和困惑度（PPL）评估

## 项目结构


```
├── cos_test.py          # 相似度评估工具 
├── g_one.py             # GLM/G系列单样本推理
├── g_test.py            # GLM/G系列模型测试
├── g4_one.py            # G4模型单样本推理
├── g4_test.py           # G4模型测试
├── ppl_test.py          # 困惑度测试工具
├── qwen_one.py          # Qwen模型单样本推理
├── qwen_test.py         # Qwen模型测试
├── qwen_vl.py           # Qwen视觉语言模型工具
├── stance_test.py       # 立场测试评估工具
├── dataset/             # 数据集目录
│   ├── finetune_with_local_images.json        # 带本地图像的微调数据
│   └── harris_finetune_with_local_images.json # Harris数据集微调文件
└── LLaVA/               # LLaVA模型相关代码
    ├── eval_dataset.py  # 数据集评估工具
    ├── eval_one.py      # 单样本评估
    ├── eval_test.py     # 测试评估
    ├── eval_weightfusion.py # 权重融合评估
    ├── predict.py       # 预测脚本
    └── requirements.txt # LLaVA依赖项
```

## 安装与设置

1. 克隆仓库
```bash
git clone https://github.com/yourusername/StanceGen.git
cd StanceGen
```

2. 安装依赖
```bash
pip install -r requirements.txt
# 如果要使用LLaVA模型，还需要安装其依赖
pip install -r LLaVA/requirements.txt
```

## 使用方法

### 批量测试

```bash
# Qwen模型测试
python qwen_test.py --dataset ./dataset/your_test_dataset.json

# 立场测试
python stance_test.py --model qwen --dataset ./dataset/your_stance_dataset.json
```

### LLaVA模型评估

```bash
# 评估LLaVA模型
cd LLaVA
python eval_test.py --model-path /path/to/model --dataset ../dataset/harris_finetune_with_local_images.json

# 权重融合评估
python eval_weightfusion.py --model-path /path/to/model --dataset ../dataset/your_dataset.json
```


## StanceGen2024数据集

StanceGen2024是一个新的数据集，包含2024年美国总统选举中的推文-图像/视频对及立场标注的用户评论。该数据集捕捉了丰富的多模态交互，并包含立场和风格的细粒度标签，使研究人员能够深入研究多模态政治内容如何塑造立场表达。

项目包含多个数据集文件，位于`dataset/`目录下：
- `harris_finetune_with_local_images.json`: 包含本地图像路径的微调数据集
- `harris_finetune_with_local_images.json`: Harris数据集的微调文件


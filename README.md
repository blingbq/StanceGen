# SDMG

SDMG is a framework for Stance-Driven Multimodal Reply Generation and Evaluation, integrating the capabilities of various large language models.

## Project Overview

SDMG aims to leverage advanced language models and multimodal models for stance-driven content generation, analysis, and evaluation. This project introduces the stance-driven multimodal reply generation task, which focuses on generating stance-consistent comments based on social media posts containing both text and visual information.

Key Features:
1. Supports multiple models, including Qwen (Tongyi Qianwen), LLaVA (Large Language and Vision Assistant), and the GPT series.
2. Provides the StanceGen2024 dataset, featuring tweet-image/video pairs and user comments with stance annotations from the 2024 U.S. presidential election.
3. Proposes the SDMG (Stance-Driven Multimodal Generation) framework, integrating multimodal feature fusion and stance guidance.
4. Supports multiple evaluation methods, including cosine similarity (COS) and perplexity (PPL).


## Project Structure


```
├── cos_test.py          # Cosine similarity evaluation tool
├── g_one.py             # Single-sample inference with GPT models
├── g_test.py            # GPT model testing script
├── g4_one.py            # Single-sample inference with GPT-4o model
├── g4_test.py           # GPT-4o model testing script
├── ppl_test.py          # Perplexity evaluation tool
├── qwen_one.py          # Single-sample inference with Qwen model
├── qwen_test.py         # Qwen model testing script
├── qwen_vl.py           # Qwen vision-language model utilities
├── stance_test.py       # Stance controllability evaluation tool
├── dataset/             # Dataset directory
│   ├── finetune_with_local_images.json        # Fine-tuning data with local images
│   └── harris_finetune_with_local_images.json # Harris dataset fine-tuning file
└── LLaVA/               # LLaVA model related scripts
    ├── eval_dataset.py  # Dataset evaluation
    ├── eval_one.py      # Single-sample evaluation
    ├── eval_test.py     # Evaluation testing script
    ├── eval_weightfusion.py # Weighted fusion evaluation
    ├── predict.py       # Prediction script
    └── requirements.txt # LLaVA dependencies
```

## Installation and Setup


1. Clone the repository
```bash
git clone https://github.com/yourusername/StanceGen.git
cd StanceGen
```

2. Install dependencies
```bash
pip install -r requirements.txt
# If you plan to use the LLaVA model, install its dependencies as well:
pip install -r LLaVA/requirements.txt
```

## Usage

### Batch Testing

```bash
# Run Qwen model testing
python qwen_test.py --dataset ./dataset/your_test_dataset.json

# Run stance controllability testing
python stance_test.py --model qwen --dataset ./dataset/your_stance_dataset.json

```

### Model Evaluation

```bash
# Evaluate LLaVA model
cd LLaVA
python eval_test.py --model-path /path/to/model --dataset ../dataset/harris_finetune_with_local_images.json

# Weighted fusion evaluation
python eval_weightfusion.py --model-path /path/to/model --dataset ../dataset/your_dataset.json

```


## StanceGen2024 Dataset

The StanceGen2024 dataset is a newly curated dataset that contains tweet-image/video pairs and stance-annotated user comments from the 2024 U.S. presidential election. It captures rich multimodal interactions and includes fine-grained stance and style labels, providing valuable resources for researchers to explore how multimodal political content shapes stance expression.

The dataset files are located in the dataset/ directory:
- `harris_finetune_with_local_images.json`: Fine-tuning file for the Trump dataset
- `harris_finetune_with_local_images.json`: Fine-tuning file for the Harris dataset


# DCFT: Domain-Aware and Co-Adaptive Feature Transformation for Domain Adaptation Few-Shot Relation Extraction

This repository contains the dataset and code implementation for the paper ​**"Domain-aware and Co-adaptive Feature Transformation for Domain Adaption Few-shot Relation Extraction"** (accepted at LREC-COLING 2024). The project addresses cross-domain few-shot relation extraction through feature space adaptation.

## Installation
```bash
# Clone this repository
git clone https://github.com/yijunL/DCFT.git
cd DCFT
pip install -r requirements.txt  # Python 3.8+ and PyTorch 1.12+ required  
```

## Dataset
The dataset for domain adaption few-shot relation extraction can be downloaded from the official GitHub project: https://github.com/thunlp/FewRel.

## Training
Run the default training script:
```bash
sh train.sh
```

## Test
```bash
python test.py
```

# 📄 Cite
If you find this repo helpful, feel free to cite us.
```
@inproceedings{liu2024domain,  
  title={Domain-aware and Co-adaptive Feature Transformation for Domain Adaption Few-shot Relation Extraction},  
  author={Liu, Yijun and Dai, Feifei and Gu, Xiaoyan and Zhai, Minghui and Li, Bo and Zhang, Meiou},  
  booktitle={Proceedings of LREC-COLING 2024},  
  pages={5275--5285},  
  year={2024}  
}  
```

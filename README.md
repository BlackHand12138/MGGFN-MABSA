# MGGFN for Multimodal Aspect-Based Sentiment Analysis

This repository contains the source code corresponding to the manuscript currently submitted to *The Visual Computer*.

## Notice
This code repository directly corresponds to the manuscript submitted to *The Visual Computer*.  
If you use this code in your research, please cite the corresponding manuscript.

## Overview
This repository provides the implementation of a multimodal aspect-based sentiment analysis (MABSA) model.

The code includes:
- the main training script
- the model implementation
- gated fusion and channel-spatial fusion modules
- image feature extraction utilities
- data loading and preprocessing utilities

## File Structure
```text
.
├── train.py
├── MyModel.py
├── Gated_Fusion.py
├── Channel_Spatial_Fusion.py
├── img_deal_by_vit.py
├── data_utils.py
├── requirements.txt
├── README.md
├── LICENSE
└── DATA_NOTICE.md
```

## Environment
Recommended environment:
- Python  3.10
- PyTorch
- torchvision
- transformers
- numpy
- pandas
- scikit-learn
- opencv-python
- tqdm

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Data Preparation
Please prepare the datasets and related files locally before training.

The code expects the following structure:

```text
data/
├── twitter2015/
│   ├── train.tsv
│   ├── dev.tsv
│   └── test.tsv
├── twitter2017/
│   ├── train.tsv
│   ├── dev.tsv
│   └── test.tsv
├── caption/
│   ├── twitter2015_images.json
│   └── twitter2017_images.json
├── face_descriptions/
│   ├── twitter2015_images_face.json
│   └── twitter2017_images_face.json
├── imgDealFile/
│   ├── twitter2015_images.pkl
│   └── twitter2017_images.pkl
├── oriAdj/
│   ├── train_ori_adj.pkl
│   ├── dev_ori_adj.pkl
│   └── test_ori_adj.pkl
└── HF/
    ├── config.json
    └── preprocessor_config.json
```

In addition, the training script may create or read cached dataloader files under:

```text
middleFile/
├── twitter15_train_datas.pkl
├── twitter15_val_datas.pkl
├── twitter15_test_datas.pkl
├── twitter17_train_datas.pkl
├── twitter17_val_datas.pkl
└── twitter17_test_datas.pkl
```

Please also create the following directory for output results:

```text
result/
```

## Running
Example commands:

Train on Twitter-2015:
```bash
python train.py --dataset twitter15 --model_name simpleBert --MAX_LEN 50 --BATCH_SIZE 32 --EPOCHS 20 --LEARNING_RATE 5e-5 --DEVICE cuda:0
```

Train on Twitter-2017:
```bash
python train.py --dataset twitter17 --model_name simpleBert --MAX_LEN 50 --BATCH_SIZE 32 --EPOCHS 20 --LEARNING_RATE 5e-5 --DEVICE cuda:0
```

## Notes
- The code uses `bert-base-uncased` for text encoding.
- The image features are extracted based on ViT-related processing.
- Please make sure all local data paths are correctly prepared before running the code.
- Some paths in the code are hard-coded. You may need to modify them according to your local environment.

## Reproducibility
To facilitate reproducibility, this repository provides:
- the source code of the model
- the training script
- dependency requirements
- dataset path conventions and preprocessing-related file descriptions


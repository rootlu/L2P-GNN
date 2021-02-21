# Learning to Pre-train Graph Neural Networks

This repository is the official implementation of AAAI-2021 paper [Learning to Pre-train Graph Neural Networks](https://yuanfulu.github.io/publication/AAAI-L2PGNN.pdf)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Dataset

All the necessary data files can be downloaded from the following links.

For Biology dataset, download from [Google Drive](https://drive.google.com/drive/folders/18vpBvSajIrme2xsbx8Oq8aTWIWMlSgJT?usp=sharing) and [BaiduYun](https://pan.baidu.com/s/1Yv6dN7F1jgTSz9-nU1eN-A) (Extraction code: j97n), unzip it, and put the  under `data/bio/`.

The new compilation of bibliographic graphs, i.e., PreDBLP,  download from [Google Drive](https://drive.google.com/drive/folders/18vpBvSajIrme2xsbx8Oq8aTWIWMlSgJT?usp=sharing) and [BaiduYun](https://pan.baidu.com/s/1Yv6dN7F1jgTSz9-nU1eN-A) (Extraction code: j97n), unzip it, and move the `dblp.graph` file to `data/dblp/unsupervised/processed/` and the `dblpfinetune.graph` file to `data/dblp/supervised/processed/`, respectively.


## Training

To pre-train L2P-GNN on Biology dataset w.r.t. GIN model, run this command:

```shell
python main.py --dataset DATASET  --gnn_type GNN_MODEL --model_file PRE_TRAINED_MODEL_NAME --device 1
```

The pre-trained models are saved into `res/DATASET/` .

## Evaluation

To fine--tune L2P-GNN on Biology dataset, run:

```shell
python eval_bio.py --dataset DATASET  --gnn_type GNN_MODEL --emb_trained_model_file EMB_TRAINED_FILE --pre_trained_model_file GNN_TRAINED_FILE --pool_trained_model_file POOL_TRAINED_FILE --result_file RESULT_FILE --device 1
```

The results w.r.t 10 random running seeds are saved into `res/DATASET/finetune_seed(0-9)/`

## Results

To analysis results of downstream tasks, run:

```shell
python result_analysis.py  --dataset DATASET --times SEED_NUM
```

where `SEED_NUM` is the number of random seed ranging from 0 to 9, thus it is usually set to 10.

## Reproducing results in the paper

Our results in the paper can be reproduced by directly running:

```shell
python eval_bio.py --dataset bio --gnn_type gin --emb_trained_model_file co_adaptation_5_300_gin_50_emb.pth --pre_trained_model_file co_adaptation_5_300_gin_50_gnn.pth --pool_trained_model_file co_adaptation_5_300_gin_50_pool.pth --result_file co_adaptation_5_300_gin_50 --device 0
```

and

```shell
python eval_dblp.py --dataset dblp --gnn_type gin --split random --emb_trained_model_file co_adaptation_5_300_s50q30_gin_20_emb.pth --pre_trained_model_file co_adaptation_5_300_s50q30_gin_20_gnn.pth --pool_trained_model_file co_adaptation_5_300_s50q30_gin_20_pool.pth --result_file co_adaptation_5_300_s50q30_gin_20  --device 0 --dropout_ratio 0.1
```


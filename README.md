# DominoSearch
This is repository for codes and models of NeurIPS2021 paper - **[DominoSearch: Find layer-wise fine-grained N:M sparse schemes from dense neural networks](https://openreview.net/forum?id=IGrC6koW_g)**

## Search:
```
git clone https://github.com/NM-sparsity/DominoSearch.git
cd DominoSearch/DominoSearch/search/script_resnet_ImageNet
```
We provide several search scripts for different sparse-ratio target, you can specify your own target and change the parameters accordingly.
Note, you need to first specify your ImageNet [dataset path](https://github.com/NM-sparsity/DominoSearch/blob/main/DominoSearch/search/script_resnet_ImageNet/configs/config_resnet50_img_mix_from_dense.yaml) 

The searching phase could take 2-3 hours, then you will get searched schemes stored in a txt file, which will be needed as input for mixed-sparsity training. 

Below is an example of output formate.

```
{'SparseConv0_3-64-(7, 7)': [16, 16], 'SparseConv1_64-64-(1, 1)': [16, 16], 'SparseConv2_64-64-(3, 3)': [4, 16], 'SparseConv3_64-256-(1, 1)': [8, 16], 'SparseConv4_64-256-(1, 1)': [8, 16], 'SparseConv5_256-64-(1, 1)': [8, 16], 'SparseConv6_64-64-(3, 3)': [4, 16], 'SparseConv7_64-256-(1, 1)': [8, 16], 'SparseConv8_256-64-(1, 1)': [8, 16], 'SparseConv9_64-64-(3, 3)': [4, 16], 'SparseConv10_64-256-(1, 1)': [8, 16], 'SparseConv11_256-128-(1, 1)': [8, 16], 'SparseConv12_128-128-(3, 3)': [2, 16], 'SparseConv13_128-512-(1, 1)': [8, 16], 'SparseConv14_256-512-(1, 1)': [4, 16], 'SparseConv15_512-128-(1, 1)': [8, 16], 'SparseConv16_128-128-(3, 3)': [4, 16], 'SparseConv17_128-512-(1, 1)': [8, 16], 'SparseConv18_512-128-(1, 1)': [8, 16], 'SparseConv19_128-128-(3, 3)': [4, 16], 'SparseConv20_128-512-(1, 1)': [8, 16], 'SparseConv21_512-128-(1, 1)': [8, 16], 'SparseConv22_128-128-(3, 3)': [2, 16], 'SparseConv23_128-512-(1, 1)': [8, 16], 'SparseConv24_512-256-(1, 1)': [4, 16], 'SparseConv25_256-256-(3, 3)': [2, 16], 'SparseConv26_256-1024-(1, 1)': [4, 16], 'SparseConv27_512-1024-(1, 1)': [4, 16], 'SparseConv28_1024-256-(1, 1)': [4, 16], 'SparseConv29_256-256-(3, 3)': [2, 16], 'SparseConv30_256-1024-(1, 1)': [4, 16], 'SparseConv31_1024-256-(1, 1)': [4, 16], 'SparseConv32_256-256-(3, 3)': [2, 16], 'SparseConv33_256-1024-(1, 1)': [4, 16], 'SparseConv34_1024-256-(1, 1)': [4, 16], 'SparseConv35_256-256-(3, 3)': [2, 16], 'SparseConv36_256-1024-(1, 1)': [4, 16], 'SparseConv37_1024-256-(1, 1)': [4, 16], 'SparseConv38_256-256-(3, 3)': [2, 16], 'SparseConv39_256-1024-(1, 1)': [4, 16], 'SparseConv40_1024-256-(1, 1)': [4, 16], 'SparseConv41_256-256-(3, 3)': [2, 16], 'SparseConv42_256-1024-(1, 1)': [4, 16], 'SparseConv43_1024-512-(1, 1)': [4, 16], 'SparseConv44_512-512-(3, 3)': [2, 16], 'SparseConv45_512-2048-(1, 1)': [4, 16], 'SparseConv46_1024-2048-(1, 1)': [2, 16], 'SparseConv47_2048-512-(1, 1)': [4, 16], 'SparseConv48_512-512-(3, 3)': [2, 16], 'SparseConv49_512-2048-(1, 1)': [4, 16], 'SparseConv50_2048-512-(1, 1)': [4, 16], 'SparseConv51_512-512-(3, 3)': [2, 16], 'SparseConv52_512-2048-(1, 1)': [4, 16], 'Linear0_2048-1000': [4, 16]}
```

## Train:
After getting the layer-wise sparse schemes, we need to fine-tune with the schemes to recover the accuracy. The training code is based on [NM-sparsity](https://github.com/NM-sparsity/NM-sparsity)/ [ICLR2021 paper](https://arxiv.org/abs/2102.04010), where we made some changes to support flexible N:M schemes. 


Below is an example of training layer-wise sparse resnet50 with 80% overall sparsity. 
```
cd DominoSearch\DominoSearch\train\classification_sparsity_level\train_imagenet
 python -m torch.distributed.launch --nproc_per_node=8 ../train_imagenet.py --config ./configs/config_resnet50.yaml  --base_lr 0.01 --decay 0.0005 --epochs 120 --schemes_file ./schemes/resnet50_M16_0.80.txt --model_dir ./resnet50/resnet50_0.80_M16
```




# Experiments

We provide the trained models of the experiments. Please check our paper for details and intepretations of the experiments.

### ResNet50 experiments in section 4.1

|  Model Name  | TOP1 Accuracy   | Trained Model  | Searched schemes | 
| ------------- |:-------------:| -----:|  -----:  |
| resnet50 - 0.80 model size |  76.7  | [google drive](https://drive.google.com/file/d/1eZ6q_XKo2yDz6F87xYPhT6GnC8eVJ9tx/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1QzPa9CWE9gOTvEH1kih7vTJR6lO8Mc1j/view?usp=sharing) |
| resnet50 - 0.875 model size      | 75.7      |  [google drive](https://drive.google.com/file/d/1YImPgGbmJtzxgGnOBsDAGlKPBO-m6YiA/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1ebKLLTZKhujaW8TQbR5ZUi3rV13TSRxk/view?usp=sharing) |
| resnet50 - 0.9375 model size | 73.5      |  [google drive](https://drive.google.com/file/d/1Q6AL6zxLW77eBw5kTx8KyhkOba5RHKP3/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1nSpBVdDTSa_jKzF-1h1t62D4tJq9Ntsx/view?usp=sharing) |
|resnet50 - 8x FLOPs| 75.4 | [google drive](https://drive.google.com/file/d/1A_WR72Y-Si1yt84H6aW5rXURnv7dX4QV/view?usp=sharing)| [google drive](https://drive.google.com/file/d/1HPs0bU8z0xh6XYpZUnzG9toKSIj6m0-Z/view?usp=sharing) |
|resnet50- 16x FLOPs| 73.4 | [google drive](https://drive.google.com/file/d/1s6Zz99bWt4_XPdZCtuqUjg_70G7zJyAr/view?usp=sharing)  | [google drive](https://drive.google.com/file/d/1dnZIwoUUpiFatDo1sxyPaJGk90yh__y4/view?usp=sharing) |

### Ablation experiments of ResNet50 in section 5.3

|  Model Name  | TOP1 Accuracy   | Trained Model  | Train log | 
| ------------- |:-------------: | -----:  | -----: |
| Ablation E3 |  76.1  | [google drive](https://drive.google.com/file/d/1i2S8Q-ely6U5i_gw35YRXbdoijKQhOBi/view?usp=sharing)  |  [google drive](https://drive.google.com/file/d/1JzrOiXLdzg8mXf0nddHEduBLNrMwLzVX/view?usp=sharing) |
| Ablation E4 |  76.4  | [google drive](https://drive.google.com/file/d/1WgNrGuT3ltCsCu3W-XP-G-pv7TfZwGYm/view?usp=sharing)  | [google drive](https://drive.google.com/file/d/1n3JrR0XuP72KU4W1D6PlAp1tSDwxoJXf/view?usp=sharing) | 
| Ablation E6 |  76.6  | [google drive](https://drive.google.com/file/d/199VE8CwGoUnvQlvD4A0nCIH1h9TbxYGD/view?usp=sharing)  | [google drive](https://drive.google.com/file/d/1ZjEKlXSEz7iNBSKPiZ4zL7QVp6tyaRuY/view?usp=sharing) | 
| Ablation E7 |  75.6  | [google drive](https://drive.google.com/file/d/1gpy0m9EZUmQsMt5pKPlOHRaJNisepGb2/view?usp=sharing)  | [google drive](https://drive.google.com/file/d/1RNDYr7soWqv7SQhHfg9dJ2s6CtWSnb61/view?usp=sharing) | 



# Citation
```
@inproceedings{
sun2021dominosearch,
title={DominoSearch: Find layer-wise fine-grained N:M sparse schemes from dense neural networks},
author={Wei Sun and Aojun Zhou and Sander Stuijk and Rob G. J. Wijnhoven and Andrew Nelson and Hongsheng Li and Henk Corporaal},
booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
year={2021},
url={https://openreview.net/forum?id=IGrC6koW_g}
}
```

# References
* [Learning N:M Fine-grained Structured Sparse Neural Networks From Scratch](https://openreview.net/forum?id=K9bw7vqp_s)

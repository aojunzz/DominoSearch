# DominoSearch
This is repository for codes and models of NeurIPS2021 paper - **DominoSearch: Find layer-wise fine-grained N:M sparse schemes from dense neural networks**

Instructions and other materials will be released soon.

## TODO:
instructions of how to use DominoSearch to find layer-wise N:M schemes
## TODO:
Instructions and codes to train sparse neural network with layer-wise N:M



# Experiments

We provide the trained models of the experiments. Please check our paper for details and intepretations of the experiments.

## ResNet50 experiments in section 4.1

|  Model Name  | TOP1 Accuracy   | Trained Model  | Searched schemes | 
| ------------- |:-------------:| -----:|  -----:  |
| resnet50 - 0.80 model size |  76.7  | [google drive](https://drive.google.com/file/d/1eZ6q_XKo2yDz6F87xYPhT6GnC8eVJ9tx/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1QzPa9CWE9gOTvEH1kih7vTJR6lO8Mc1j/view?usp=sharing) |
| resnet50 - 0.875 model size      | 75.7      |  [google drive](https://drive.google.com/file/d/1YImPgGbmJtzxgGnOBsDAGlKPBO-m6YiA/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1ebKLLTZKhujaW8TQbR5ZUi3rV13TSRxk/view?usp=sharing) |
| resnet50 - 0.9375 model size | 73.5      |  [google drive](https://drive.google.com/file/d/1Q6AL6zxLW77eBw5kTx8KyhkOba5RHKP3/view?usp=sharing) | [google drive](https://drive.google.com/file/d/1nSpBVdDTSa_jKzF-1h1t62D4tJq9Ntsx/view?usp=sharing) |
|resnet50 - 8x FLOPs| 75.4 | [google drive](https://drive.google.com/file/d/1A_WR72Y-Si1yt84H6aW5rXURnv7dX4QV/view?usp=sharing)| [google drive](https://drive.google.com/file/d/1HPs0bU8z0xh6XYpZUnzG9toKSIj6m0-Z/view?usp=sharing) |
|resnet50- 16x FLOPs| 73.4 | [google drive](https://drive.google.com/file/d/1s6Zz99bWt4_XPdZCtuqUjg_70G7zJyAr/view?usp=sharing)  | [google drive](https://drive.google.com/file/d/1dnZIwoUUpiFatDo1sxyPaJGk90yh__y4/view?usp=sharing) |

## Ablation experiments of ResNet50 in section 5.3

|  Model Name  | TOP1 Accuracy   | Trained Model  | 
| ------------- |:-------------:| -----:|  
| Ablation E3 |  76.1  | [google drive](https://drive.google.com/file/d/1i2S8Q-ely6U5i_gw35YRXbdoijKQhOBi/view?usp=sharing)  |
| Ablation E4 |  76.4  | [google drive](https://drive.google.com/file/d/1WgNrGuT3ltCsCu3W-XP-G-pv7TfZwGYm/view?usp=sharing)  |
| Ablation E6 |  76.6  | [google drive](https://drive.google.com/file/d/199VE8CwGoUnvQlvD4A0nCIH1h9TbxYGD/view?usp=sharing)  |
| Ablation E7 |  75.6  | [google drive](https://drive.google.com/file/d/1gpy0m9EZUmQsMt5pKPlOHRaJNisepGb2/view?usp=sharing)  |
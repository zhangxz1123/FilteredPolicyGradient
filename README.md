TRPO
https://user-images.githubusercontent.com/18234564/109558347-e7f5d000-7a9e-11eb-9b22-57c75c89baf1.mp4

FPG
https://user-images.githubusercontent.com/18234564/109558352-eaf0c080-7a9e-11eb-9921-2c5446288711.mp4

# PyTorch implementation of the Filtered Policy Gradient (FPG) algorithm

This is a PyTorch implementation of ["Filtered Policy Gradient (FPG)"](https://arxiv.org/abs/2102.05800). Please make sure to install the necessary dependencies, particularly Pytorch and MuJoCo.

The current version of FPG is using Gaussian policies suited for continuous control problems. Minimum changes are required to work with discrete action space (log probability functions, etc).

## Usage

```
python main.py --env-name "Swimmer-v3" --sever 0 --attack_norm 10 --max_iter_num 200 --eps 0.01 # Vanilla TRPO
python main.py --env-name "Swimmer-v3" --sever 1 --attack_norm 10 --max_iter_num 200 --eps 0.01 # FPG
```

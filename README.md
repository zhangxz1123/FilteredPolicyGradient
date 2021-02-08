# PyTorch implementation of the Filtered Policy Gradient (FPG) algorithm

##

This is a PyTorch implementation of ["Filtered Policy Gradient (FPG)"](https://arxiv.org/abs/1502.05477). Please make sure to install the necessary dependencies, particularly Pytorch and MuJoCo.

The current version of FPG is using Gaussian policies suited for continuous control problems. Minimum changes are required to work with discrete action space (log probability functions, etc).

## Usage

```
python main.py --env-name "Swimmer-v3" --sever 0 --attack_norm 10 --max_iter_num 200 --eps 0.01 # Vanilla TRPO
python main.py --env-name "Swimmer-v3" --sever 1 --attack_norm 10 --max_iter_num 200 --eps 0.01 # FPG
```

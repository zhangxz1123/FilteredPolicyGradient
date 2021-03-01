# PyTorch implementation of the Filtered Policy Gradient (FPG) algorithm

This is a PyTorch implementation of ["Filtered Policy Gradient (FPG)"](https://arxiv.org/abs/2102.05800). Please make sure to install the necessary dependencies, particularly Pytorch and MuJoCo.

The current version of FPG is using Gaussian policies suited for continuous control problems. Minimum changes are required to work with discrete action space (log probability functions, etc).

## Usage

```
python main.py --env-name "Swimmer-v3" --sever 0 --attack_norm 10 --max_iter_num 200 --eps 0.01 # Vanilla TRPO
python main.py --env-name "Swimmer-v3" --sever 1 --attack_norm 10 --max_iter_num 200 --eps 0.01 # FPG
```

TRPO is fooled to learn the backward running policy on HalfCheetah with epsilon=0.01 and delta large enough.

![cheetah_backward](https://user-images.githubusercontent.com/18234564/109559118-e7116e00-7a9f-11eb-8f6e-06daf3eb8ebf.gif)

FPG remains unaffected.

![cheetah_forward](https://user-images.githubusercontent.com/18234564/109559179-fabcd480-7a9f-11eb-9459-51c423d0ecaa.gif)

# HOLD Reward Models

Official implementation of the following paper:  

<p align="center"><b>Learning Reward Functions for Robotic Manipulation by Observing Humans</b><br>
Minttu Alakuijala, Gabriel Dulac-Arnold, Julien Mairal, Jean Ponce, Cordelia Schmid<br>
ICRA 2023<br>
<a href="https://arxiv.org/abs/2211.09019">[Paper]</a> | <a href="https://sites.google.com/view/hold-rewards">[Project website]</a></p>

This repository includes the training of HOLD models (a.k.a functional distance models) on video data.
This implementation is based on [Scenic](https://github.com/google-research/scenic).

For the RL policy training experiments in the paper, see [https://github.com/minttusofia/hold-policies](https://github.com/minttusofia/hold-policies).
  

## Installation

To train models, **python 3.9** is required (required for dmvr, which is a dependency of scenic).

To install this codebase, run
```shell
$ git clone https://github.com/minttusofia/hold-rewards.git
$ cd hold-rewards
$ pip install .
```
For a GPU-enabled installation of jax (recommended), see [https://github.com/google/jax/#pip-installation-gpu-cuda-installed-via-pip-easier](https://github.com/google/jax/#pip-installation-gpu-cuda-installed-via-pip-easier).


## Training HOLD on Something-Something v2

Make the following changes to the scenic config file  
(in `scenic/projects/func_dist/configs/holdr/vivit_large_factorized_encoder.py` for HOLD-R, or  
`scenic/projects/func_dist/configs/holdc/resnet50.py` for HOLD-C):
* Set `DATA_DIR` to the directory where Something-Something v2 data (and optionally, any pretrained model checkpoints) are saved.
* Set `NUM_DEVICES` to the number of GPUs / TPUs to use.

Run
```shell
python scenic/projects/func_dist/main.py \
--config=scenic/projects/func_dist/configs/holdr/vivit_large_factorized_encoder.py \
--workdir=/PATH/TO/OUT_DIR
```
where `/PATH/TO/OUT_DIR` is the directory to which experiment checkpoints will be written.

For HOLD-C, use `--config=scenic/projects/func_dist/configs/holdc/resnet50.py`.


## Trained models

We release the trained model checkpoints used in the paper:  
* [HOLD-C](https://huggingface.co/minttusofia/hold/tree/main/holdc)  
* [HOLD-R](https://huggingface.co/minttusofia/hold/tree/main/holdr)

To use these as reward models in policy training with SAC (as in the paper), please refer to our policy training repo [https://github.com/minttusofia/hold-policies](https://github.com/minttusofia/hold-policies).

 
## Citing HOLD

If you found this implementation or the released models useful, you are encouraged to cite our paper:
```bibtex
@article{alakuijala2023learning,  
    title={Learning Reward Functions for Robotic Manipulation by Observing Humans},  
    author={Alakuijala, Minttu and Dulac-Arnold, Gabriel and Mairal, Julien and Ponce, Jean and Schmid, Cordelia},  
    journal={2023 IEEE International Conference on Robotics and Automation (ICRA)},  
    year={2023},  
}
```

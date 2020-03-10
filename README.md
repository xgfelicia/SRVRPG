# SRVRPG
Stochastic Recursive Variance Reduced Policy Gradient

ARXIV: [Sample Efficient Policy Gradient Methods with Recursive Variance Reduction](https://arxiv.org/pdf/1909.08610.pdf)

## Includes: 
- SRVR-PG implementation in rllab 
- some setup files for reference (used on Ubuntu 16.04) 

## To-do: 
- add PyTorch implementation 


### Setup Troubleshooting Q&A 
* Cannot setup environment using _environment.yml_
  - Remove dependency `chainer==1.18.0` from _environment.yml_  
* Cannot find libSDL: `ImportError: libSDL_ttf-2.0.so.0: cannot open shared object file: No such file or directory`
  - Install missing dependency `sudo apt-get install libsdl-ttf2.0-0` on Ubuntu  
* `ImportError: cannot import name 'MemmapingPool'`
  - Open file `rllab/sampler/stateful_pool.py` and change the import from `from joblib.pool import MemmapingPool` to `from joblib.pool import MemmappingPool`
  - Issue: https://github.com/rll/rllab/issues/240
* No such file `...rllab/envs/box2d/models/cartpole.xml.mako`
  - Run setup install again with `misc-files/setup.py` and `misc-files/MANIFEST.in`

### Additional code sources: 
- Papini, M., Binaghi, D., Canonaco, G., Pirotta, M., & Restelli, M. (2018). Stochastic Variance-Reduced Policy Gradient. _ICML._
- SVRPG original implementation: https://github.com/Dam930/rllab

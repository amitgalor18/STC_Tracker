#!/usr/bin/env bash
TORCH_CUDA_ARCH_LIST="3.5;3.7;5.0;5.2;6.0+PTX;6.1+PTX;7.0+PTX;7.5+PTX;6.1;7.0;7.5;8.0;8.0+PTX;8.6"
python setup.py build develop

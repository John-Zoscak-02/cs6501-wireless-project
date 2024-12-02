#!/bin/bash

pip install -e /home/jmz9sad/MVDNet/detectron2 
pip install -e .
python ./tools/train.py --config ./configs/train_config.yaml

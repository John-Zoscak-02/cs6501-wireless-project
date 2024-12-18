#!/bin/bash

pip install -r requirements.txt
pip install -e detectron2/ 
pip install -e .
python ./tools/train.py --config ./configs/train_config.yaml

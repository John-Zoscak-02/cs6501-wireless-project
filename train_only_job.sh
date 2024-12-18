#!/bin/bash

pip install -e detectron2/ 
pip install -e .
pip install pycocotools
python ./tools/eval.py --config ./configs/eval_config.yaml

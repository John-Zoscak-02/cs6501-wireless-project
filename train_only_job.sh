#!/bin/bash

pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -e /home/jmz9sad/cs6501-wireless-project/detectron2 
pip install -e .
python ./tools/train.py --config ./configs/train_config.yaml

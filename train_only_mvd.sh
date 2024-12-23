#!/bin/bash

#SBATCH --time=6:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=8   # number of CPU cores per task
#SBATCH --gres=gpu:a6000:1   # gpu devices per node
#SBATCH --partition gpu   # partition
#SBATCH --mem-per-cpu=16G   # memory per CPU core
#SBATCH -J "train_only_mvd"   # job name
#SBATCH --account=cs_6501_ws4iot   # allocation name

module load apptainer conda cuda/12.4.1 cudnn

# # >>> conda initialize >>>
# # !! Contents within this block are managed by 'conda init' !!
# __conda_setup="$('/sfs/applications/202406_build/software/standard/core/miniforge/24.3.0-py3.11/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/sfs/applications/202406_build/software/standard/core/miniforge/24.3.0-py3.11/etc/profile.d/conda.sh" ]; then
#         . "/sfs/applications/202406_build/software/standard/core/miniforge/24.3.0-py3.11/etc/profile.d/conda.sh"
#     else
#         export PATH="/sfs/applications/202406_build/software/standard/core/miniforge/24.3.0-py3.11/bin:$PATH"
#     fi
# fi
# unset __conda_setup
# # <<< conda initialize <<<
# 
# conda activate mvdnet3
# 

apptainer run --nv apptainer/mvdnet4.sif ./train_only_job.sh
# pip install -r requirements.txt
# pip install -e detectron2
# pip install -e .
# python ./tools/train.py --config ./configs/train_config.yaml

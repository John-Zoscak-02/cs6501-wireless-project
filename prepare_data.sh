#!/bin/bash

#SBATCH --time=6:00:00   # job time limit
#SBATCH --nodes=2   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=8   # number of CPU cores per task
#SBATCH --gres=gpu:1   # gpu devices per node
#SBATCH --partition gpu   # partition
#SBATCH --mem-per-cpu=6G   # memory per CPU core
#SBATCH -J "prepare_data"   # job name
#SBATCH --account=cs_6501_ws4iot   # allocation name


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load cuda/11.8.0 anaconda

# >>> initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/sfs/applications/202406_build/software/standard/core/miniforge/24.3.0-py3.11/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/sfs/applications/202406_build/software/standard/core/miniforge/24.3.0-py3.11/etc/profile.d/conda.sh" ]; then
        . "/sfs/applications/202406_build/software/standard/core/miniforge/24.3.0-py3.11/etc/profile.d/conda.sh"
    else
        export PATH="/sfs/applications/202406_build/software/standard/core/miniforge/24.3.0-py3.11/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate mvdnet

cd ~/MVDNet

# Prepare Data: 
python data/sdk/prepare_radar_data.py --data_path /scratch/jmz9sad/2019-01-10-11-46-21-radar-oxford-10k  --image_size 320 --resolution 0.2
python data/sdk/prepare_lidar_data.py --data_path /scratch/jmz9sad/2019-01-10-11-46-21-radar-oxford-10k 
python data/sdk/prepare_fog_data.py --data_path /scratch/jmz9sad/2019-01-10-11-46-21-radar-oxford-10k --beta 0.05
 

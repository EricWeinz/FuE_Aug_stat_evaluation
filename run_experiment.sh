#!/bin/bash

#SBATCH --no-requeue
#SBATCH --partition=alpha
#SBATCH --nodes=1                   
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --mincpus=1
#SBATCH --time=72:00:00                             
#SBATCH --job-name=alpha2_vgg_new_cifar10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eric.weinzierl@mailbox.tu-dresden.de
#SBATCH --output=output-%x.out

module --force purge                          				
module load modenv/hiera CUDA/11.7.0 GCCcore/11.3.0 Python/3.10.4
pip install torch torchvision tensorboardX baal tqdm Pillow

if [ -d "/scratch/ws/0/erwe517e-train_sec_ws_run_vgg" ] 
then
    echo "Workspace exists and will be used"
    source /scratch/ws/0/erwe517e-ws_run_vgg_new/pyenv/bin/activate
else
    WS_NAME="train_sec_ws_run_vgg"
    FS_NAME="scratch"
    DURATION=10

    echo "Creating new environment $WS_NAME in FS $FS_NAME for $DURATION"
    WS_PATH=$(ws_allocate -F $FS_NAME $WS_NAME $DURATION)
    virtualenv $WS_PATH/pyenv
    source $WS_PATH/pyenv/bin/activate
    pip install --upgrade pip
    pip install torch torchvision tensorboard
    pip install baal tqdm Pillow
fi

python vgg_augmented_cifar10_orgsize.py
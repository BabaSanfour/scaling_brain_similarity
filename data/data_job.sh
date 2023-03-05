#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca
#SBATCH --mail-type=ALL
#SBATCH --time=1:00:00        
#SBATCH --mem=16   
#SBATCH --output=S-subtrain-%j.out
#SBATCH --error=S-subtrain-%j.err

source /home/hamza97/venv/bin/activate


cd $SLURM_TMPDIR
mkdir work
cd work
unzip /home/hamza97/scratch/data/imagenet-object-localization-challenge.zip -d .

python /home/hamza97/scaling_brain_similarity/data/train_subsamples.py

# The computations are done, so clean up the data set...
cd $SLURM_TMPDIR
tar -cf /home/hamza97/scratch/data/new_imagenet_subtrain.tar work

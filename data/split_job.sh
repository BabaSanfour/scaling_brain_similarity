#!/bin/bash
#SBATCH --account=def-kjerbi
#SBATCH --mail-user=hamza.abdelhedi@umontreal.ca
#SBATCH --mail-type=ALL
#SBATCH --time=1:00:00        
#SBATCH --mem=64G   
#SBATCH --output=Toh5-%j.out
#SBATCH --error=Toh5-%j.err

source /home/hamza97/venv/bin/activate


cd $SLURM_TMPDIR
mkdir work
cd work
tar -xf /home/hamza97/scratch/data/scaling_data/1.tar

python /home/hamza97/scaling_brain_similarity/data/ImageNet_train_data_to_h5.py
#!/bin/bash -l
#SBATCH -J las460fc
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=user_id@student.uni.lu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -G 1
#SBATCH --time=2-00:00:00
#SBATCH -p gpu
#SBATCH --qos=normal
#SBATCH --output='output/las-fixed-100-character%A.out'

echo "== Starting run at $(date)"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir. : ${SLURM_SUBMIT_DIR}"

source .venv/bin/activate
python3 main.py --config config/libri/las_460_fixed_character.yaml --logdir las_460_fixed_character_logdir --name las_460_fixed_character

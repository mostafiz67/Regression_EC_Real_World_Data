#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --mem=10000M  # memory
#SBATCH --cpus-per-task=16
#SBATCH --output=Regression-EC-RL-Data-%j.out  # %N for node name, %j for jobID
#SBATCH --time=00-00:20     # time (DD-HH:MM)
#SBATCH --mail-user=x2020fpt@stfx.ca # used to send emailS
#SBATCH --mail-type=ALL

module load python/3.8
SOURCEDIR=/home/x2020fpt/projects/def-jlevman/x2020fpt/

source /home/x2020fpt/projects/def-jlevman/x2020fpt/Regression-EC-RL-Data/.venv/bin/activate && echo "$(date +"%T"):  Activated python virtualenv"

echo "$(date +"%T"):  start running model!"
python3 /home/x2020fpt/projects/def-jlevman/x2020fpt/Regression-EC-RL-Data/run.py --dataset=Breast-Cancer --progress
echo "$(date +"%T"):  Finished running!"


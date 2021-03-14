#!/bin/bash
# Define the job name
#SBATCH --job-name=RN_BIM_TEST
#SBATCH --output=prints/pn_bim_test.out
#SBATCH --time=5-0:0:0
# Requested number of cores. Choose either of, or both of

export CUDA_VISIBLE_DEVICES=0
newgrp gaa
module load miniconda/2.7
source activate /home/proyectos/ada2/dhernand/tensorflow

mkdir running_prints

python AIP_ns.py 0 1.0 2 bim_data.txt > running_prints/print_1.0.txt ;
python AIP_ns.py 0 0.0001 2 bim_data.txt > running_prints/print_0.0001.txt ;




#!/bin/bash

#SBATCH --job-name=BIM_0.5_0_bim_data.txt
#SBATCH --output=exe/0.5/alpha_0.5_split_0.out

#SBATCH --time=5-0:0:0

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000

date

echo IPs - Datos sintéticos bimodales
echo alpha = 0.5
echo split = 0
echo dataset = bim_data.txt

cd ../

python3 AIP_bnn.py 0 0.5 2 bim_data.txt

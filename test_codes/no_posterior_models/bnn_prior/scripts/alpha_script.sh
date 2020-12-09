#!/bin/bash

#SBATCH --job-name=FB_DATA_ALPHA
#SBATCH --output=exe/ALPHA/alpha_ALPHA_split_SPLIT.out

#SBATCH --time=5-0:0:0

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000

date

echo IPs - Datos sintéticos DATA 
echo alpha = ALPHA
echo split = SPLIT
echo dataset = DATA

cd ../

python3 AIP_bnn.py SPLIT ALPHA 2 DATA

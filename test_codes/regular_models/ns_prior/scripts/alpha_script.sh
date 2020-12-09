#!/bin/bash

#SBATCH --job-name=RN_DATA_ALPHA
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

module load miniconda/3.6
source activate /home/proyectos/ada2/dhernand/tensorflow
export LD_LIBRARY_PATH=/home/dhernand/local/lib:/home/dhernand/local/lib64:/usr/lib:/usr/local/slurm/slurm/libls

python3 AIP_ns.py SPLIT ALPHA 2 DATA

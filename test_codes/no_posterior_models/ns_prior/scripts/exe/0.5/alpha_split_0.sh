#!/bin/bash

#SBATCH --job-name=forced_0.5_0_bim_data.txt
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

module load miniconda/3.6
source activate /home/proyectos/ada2/dhernand/tensorflow
export LD_LIBRARY_PATH=/home/dhernand/local/lib:/home/dhernand/local/lib64:/usr/lib:/usr/local/slurm/slurm/libls

python3 AIP_ns.py 0 0.5 2 bim_data.txt

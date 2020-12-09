#!/bin/bash

#SBATCH --job-name=RN_bim_data.txt_1.0
#SBATCH --output=exe/1.0/alpha_1.0_split_0.out

#SBATCH --time=5-0:0:0

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000

date

echo IPs - Datos sintéticos bim_data.txt
echo alpha = 1.0
echo split = 0
echo dataset = bim_data.txt

cd ../

module load miniconda/3.6
source activate /home/proyectos/ada2/dhernand/tensorflow
export LD_LIBRARY_PATH=/home/dhernand/local/lib:/home/dhernand/local/lib64:/usr/lib:/usr/local/slurm/slurm/libls

python3 AIP_ns.py 0 1.0 2 bim_data.txt

#!/bin/bash

#SBATCH --job-name=vip_synth_DATA
#SBATCH --output=exe/vip.out

#SBATCH --time=5-0:0:0

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2000

date

echo TEST DE IMPLICIT PROCESSES
echo dataset = DATA

cd ../
module load miniconda/3.6
newgrp ada2
source activate /home/proyectos/ada2/simonrs/anaconda/vip

python3 Main.py DATA 


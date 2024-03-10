#!/bin/bash

#------- Job Description -------

#SBATCH --job-name='ml predictions'
#SBATCH --comment='training MolE'

#------- Parametrization -------

#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=bqhn01
#SBATCH --time=10-0:0:0

#------- Input/Output -------

#SBATCH --output="/home/roberto_olayo/MolE/logdir/pretrain/%x-%j.out"
#SBATCH --error="/home/roberto_olayo/MolE/logdir/pretrain/%x-%j.err"

#------- Command -------
python representation.py
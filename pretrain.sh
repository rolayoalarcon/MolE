#!/bin/bash

#------- Job Description -------

#SBATCH --job-name='pretrain_ginconcat_100K'
#SBATCH --comment='training MolE'

#------- Parametrization -------

#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=100
#SBATCH --partition=bqhn01
#SBATCH --time=10-0:0:0

#------- Input/Output -------

#SBATCH --output="logdir/pretrain/%x-%j.out"
#SBATCH --error="logdir/pretrain/%x-%j.err"

#------- Command -------
python pretrain.py
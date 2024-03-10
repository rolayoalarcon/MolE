#!/bin/bash

#------- Job Description -------

#SBATCH --job-name='benchmarking'
#SBATCH --comment='Benchmarking'

#------- Parametrization -------

#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --partition=bqhn01
#SBATCH --time=10-0:0:0

#------- Input/Output -------

#SBATCH --output="./logdir/finetune/%x-%j.out"
#SBATCH --error="./logdir/finetune/%x-%j.err"

#------- Command -------

python finetune.py -c config_finetune.yaml -t ClinTox
#python finetune.py -c config_finetune.yaml -t BBBP
#python finetune.py -c config_finetune.yaml -t BACE
##python finetune.py -c config_finetune.yaml -t SIDER
#python finetune.py -c config_finetune.yaml -t FreeSolv
#python finetune.py -c config_finetune.yaml -t HIV 
#python finetune.py -c config_finetune.yaml -t Tox21
#python finetune.py -c config_finetune.yaml -t ESOL
#python finetune.py -c config_finetune.yaml -t Lipo
#python finetune.py -c config_finetune.yaml -t qm7
# python finetune.py -c config_finetune.yaml -t qm8
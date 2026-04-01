#!/bin/bash
#SBATCH -J hanseg_deformcl_Mixed_SOTA
#SBATCH -p gpu_4090
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G
#SBATCH --time=72:00:00
#SBATCH --comment=BASE
#SBATCH -o out/deformcl%j.out
#SBATCH -e error/deformcl%j.err

#SBATCH --gres=gpu:4

sh run_hanseg_deformcl_Mixed_SOTA.sh
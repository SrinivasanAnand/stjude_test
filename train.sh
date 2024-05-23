#!/bin/bash
#BSUB -P testVGG16
#BSUB -q gpu
#BSUB -n 2
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=5GB]"
#BSUB -gpu "num=2/host"

module load pytorch/2.2.1
pip install torchvision
python3 vgg16train.py &> out.txt

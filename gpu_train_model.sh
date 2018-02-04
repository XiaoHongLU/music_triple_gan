#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH --gres=gpu:1 # use 4 GPU
#SBATCH --mem=16000  # memory in Mb
#SBATCH -o /home/s1679450/out.txt  # send stdout to outfile
#SBATCH -e /home/s1679450/err.txt  # send stderr to errfile
#SBATCH -t 10:50:00  # time requested in hour:minute:seconds

# Setup CUDA and CUDNN related paths
export CUDA_HOME=/opt/cuda-8.0.44

export CUDNN_HOME=/opt/cuDNN-6.0_8.0

export STUDENT_ID=s1679450

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

# Setup a folder in the very fast scratch disk which can be used for storing experiment objects and any other files 
# that may require storage during execution.
mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/


# Activate the relevant virtual environment:

source /home/${STUDENT_ID}/miniconda3/bin/activate mlp


# Run the python script that will train our network
for unit_num in 100 200 300;
do
	for layer_num in 1 2 3;
	do
		export TASK_TAG=layer_${layer_num}_unit_${unit_num}
		export WORKDIR=/home/${STUDENT_ID}/music_triple_gan/Models/${TASK_TAG}
		mkdir -p ${WORKDIR}
		python /home/s1679450/music_triple_gan/lstm_keras.py ${WORKDIR} 300 ${unit_num} ${layer_num}
	done 
done


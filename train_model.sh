#!/bin/bash

JOB_NAME=$1
PY_FILE=$2
PY_ARGS=${@:3}

LOG_DIR=/home/xudong99/scratch/n_imagenet/checkpoint/"$JOB_NAME"
# set up log output folder
mkdir -p $LOG_DIR

DATETIME=$(date "+%Y-%m-%d_%H:%M:%S")
LOG_FILE=$LOG_DIR/${DATETIME}.log

PATH_TO_REPOSITORY=/home/xudong99/scratch/n_imagenet

export PYTHONPATH="$PATH_TO_REPOSITORY":$PYTHONPATH
cd "$PATH_TO_REPOSITORY"/real_cnn_model

# write to new file
# SBATCH --nodes=1
echo "#!/bin/bash
#SBATCH --gres=gpu:p100:1
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --mem=127000M
#SBATCH --time=36:00:00
#SBATCH --account=def-gigor
#SBATCH --output=$LOG_FILE
#SBATCH --error=$LOG_FILE
#SBATCH --mail-user=liuxd1215@gmail.com
#SBATCH --mail-type=ALL
echo \$SLURM_JOB_ID >> $LOG_FILE                      # log the job id
echo \$SLURM_JOB_PARTITION >> $LOG_FILE               # log the job partition
python --version >> $LOG_FILE                        # log Python version
gcc --version >> $LOG_FILE                           # log GCC version
nvcc --version >> $LOG_FILE                          # log NVCC version
nvidia-smi >> $LOG_FILE                          # log NVCC version
python $PY_FILE $PY_ARGS >> $LOG_FILE                # the script above, with its standard output appended log file
" >> ./run-${JOB_NAME}.slrm

# run the created file
sbatch run-${JOB_NAME}.slrm

# delete it
sleep 1
rm -f run-${JOB_NAME}.slrm

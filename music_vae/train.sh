#!/bin/bash
#SBATCH --account=sdbarton
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=24g
#SBATCH -J "GrooveVAE_Train"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH -C "A100|V100|A30|H100|H200|P100|L40S"
#SBATCH --array=1-100

module purge
module load cuda/12.8
module load python/3.8.13

export TMPDIR=/home/pmtirch/groovebox/tmp

DIR=$1
PARAMS_OFFSET=$2

PARAMS_FILE="${DIR}/params"
RESULTS_FILE="${DIR}/results"
RUNLOG_FILE="${DIR}/runlog"

if [ -z "$PARAMS_OFFSET" ]
then
    PARAMS_OFFSET=0
fi

if [ ! -d "$DIR" -o ! -f "$PARAMS_FILE" ]
then
    echo "Usage: $0 DIR [PARAMS_OFFSET]"
    echo "where DIR is a directory containing a file 'params' with the parameters."
    exit 1
fi

PARAMS_ID=$(( $SLURM_ARRAY_TASK_ID + $PARAMS_OFFSET ))
JOB_NAME="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

mkdir -p "$DIR"
touch "$RUNLOG_FILE"

echo "$PARAMS_ID|$JOB_NAME|$SLURM_SUBMIT_DIR" >> $RUNLOG_FILE

PARAMS=$(sed -n "${PARAMS_ID}p" ${PARAMS_FILE})

if [ -z "$PARAMS" ]; then
    echo "No params found for line ${PARAMS_ID}"
    exit 1
fi

echo "*** TRAIN ***"

TOTAL_STEPS=410000

TOTAL_STEPS=410000
RUN_DIR="/home/pmtirch/groovebox/run/${JOB_NAME}"
EVENT_DIR="${RUN_DIR}/train"

mkdir -p "$RUN_DIR"
mkdir -p "$EVENT_DIR"

progress_bar() {
    while true; do
        if ls $EVENT_DIR/events.out.tfevents.* 1>/dev/null 2>&1; then
            STEP=$(grep -a "global_step" $EVENT_DIR/events.out.tfevents.* | tail -1 | grep -oP 'global_step=\K[0-9]+')
            if [ ! -z "$STEP" ]; then
                PCT=$(( STEP * 100 / TOTAL_STEPS ))
                scontrol update JobId=$SLURM_JOB_ID JobName="VAE_${PCT}%"
            fi
        fi
        sleep 30
    done
}
progress_bar &

python music_vae_train.py \
--config=groovae_2bar_groovebox \
--run_dir=${RUN_DIR} \
--mode=train \
--tfds_name=groove/2bar-midionly \
--hparams="${PARAMS}" \
> ${RUN_DIR}/train_log.txt 2>&1

# exit if training failed
if [ $? -ne 0 ]; then
    echo "Training failed."
    exit 1
fi

echo "Training complete."
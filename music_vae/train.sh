#!/bin/bash
#SBATCH --account=sdbarton
#SBATCH --partition=quick

# export TMPDIR=/scratch/<project>/tmp

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

echo "$PARAMS_ID|$JOB_NAME|$SLURM_SUBMIT_DIR" >> $RUNLOG_FILE

PARAMS=$(tail -n +${PARAMS_ID} ${PARAMS_FILE} | head -n 1)

echo "*** TRAIN ***"
start_seconds = $SECONDS

music_vae_train \
--config=groovae_2bar_groovebox \
--run_dir=/tmp/groovebox/ \
--mode=train \
--tfds_name=groove/2bar-midionly

end_seconds = $SECONDS

duration=$((end_seconds - start_seconds))
echo "Execution time: $duration seconds"

# exit if training failed
test $? -ne 0 && exit 1

echo "*** TEST ***"
# we assembled the needed data to a single line in $TMPFILE
TMPFILE=$(mktemp)
echo -n "$PARAMS_ID|$PARAMS|$JOB_NAME|$BN|" > $TMPFILE

myprog_eval.py ${MODEL_FILE} | tr '\n\t' '| ' >> $TMPFILE
echo >> $TMPFILE

# only at the end we append it to the results file
cat $TMPFILE >> $RESULTS_FILE

# cleanup
rm $TMPFILE
rm ${MODEL_FILE}
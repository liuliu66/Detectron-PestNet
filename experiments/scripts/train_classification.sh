set -x
set -e

NET=$1

LOG="experiments/logs/classification/classification_end2end_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python2 tools/train_net.py \
    --cfg experiments/cfgs/e2e_classification_${NET}.yaml \
    OUTPUT_DIR experiments/output/classification/${NET}
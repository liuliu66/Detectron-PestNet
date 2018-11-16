set -x
set -e

NET=$1

LOG="experiments/logs/mask_rcnn/mask_rcnn_end2end_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python2 tools/train_net.py \
    --cfg experiments/cfgs/e2e_mask_rcnn_${NET}.yaml \
    --multi-gpu-testing \
    OUTPUT_DIR experiments/output/mask_rcnn/${NET}
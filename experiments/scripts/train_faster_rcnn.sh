set -x
set -e

#GPU_NMS=$1
NET=$1

LOG="experiments/logs/faster_rcnn/faster_rcnn_end2end_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python2 tools/train_net.py \
    --cfg experiments/cfgs/e2e_faster_rcnn_${NET}.yaml \
    --multi-gpu-testing \
    OUTPUT_DIR experiments/output/faster_rcnn/${NET}
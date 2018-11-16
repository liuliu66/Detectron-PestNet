set -x
set -e

#GPU_NMS=$1
NET=$1

LOG="experiments/logs/cascade_rcnn_fpn/cascade_rcnn_fpn_end2end_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python2 tools/train_net.py \
    --cfg experiments/cfgs/e2e_cascade_rcnn_${NET}-FPN.yaml \
    --multi-gpu-testing \
    OUTPUT_DIR experiments/output/cascade_rcnn_fpn/${NET}
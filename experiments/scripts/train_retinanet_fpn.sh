set -x
set -e

GPU_NMS=$1
NET=$2

LOG="experiments/logs/retinanet_end2end_${NET}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python2 tools/train_net.py \
    --cfg experiments/cfgs/e2e_${NET}-FPN.yaml
    --multi-gpu-testing \
    NUM_GPUS ${GPU_NMS} \
    USE_NCCL True \
    OUTPUT_DIR experiments/output/retinanet
set -x
set -e

GPU_NMS=$1
NET=$2

LOG="experiments/logs/retinanet_end2end_${NET}_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python2 tools/test_net.py \
    --cfg experiments/cfgs/e2e_retinanet_${NET}-FPN.yaml \
    --multi-gpu-testing \
    TEST.WEIGHTS experiments/output/retinanet/train/coco_2014_train/generalized_rcnn\model_final.pkl \
    NUM_GPUS ${GPU_NMS} \
    USE_NCCL True
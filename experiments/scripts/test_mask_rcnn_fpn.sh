set -x
set -e

GPU_NMS=$1
NET=$2

LOG="experiments/logs/mask_rcnn_end2end_${NET}_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python2 tools/test_net.py \
    --cfg experiments/cfgs/e2e_mask_rcnn_${NET}-FPN.yaml \
    --multi-gpu-testing \
    TEST.WEIGHTS experiments/output/mask_rcnn_fpn/inception-v1/train/coco_chimeibing_single_trainval/generalized_rcnn/model_final.pkl \
    NUM_GPUS ${GPU_NMS} \
    USE_NCCL True
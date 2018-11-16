set -x
set -e

NET=$1

LOG="experiments/logs/faster_rcnn_fpn_end2end_${NET}_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python2 tools/test_net_2stage.py \
    --cfg experiments/cfgs/e2e_faster_rcnn_${NET}-FPN.yaml \
    --multi-gpu-testing \
    TEST.WEIGHTS experiments/output/faster_rcnn_fpn/${NET}/train/coco_chimeibing_single_class2_trainval/generalized_rcnn/model_final.pkl \
    OUTPUT_DIR experiments/output/faster_rcnn_fpn/${NET}
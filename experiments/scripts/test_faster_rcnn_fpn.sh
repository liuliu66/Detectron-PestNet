set -x
set -e

NET=$1

LOG="experiments/logs/faster_rcnn_end2end_${NET}_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python2 tools/test_net.py \
    --cfg experiments/cfgs/e2e_faster_rcnn_${NET}-FPN.yaml \
    --multi-gpu-testing \
    TEST.WEIGHTS experiments/output/faster_rcnn/${NET}/train/coco_jiaduo_trainval/generalized_rcnn/model_iter199999.pkl \
    OUTPUT_DIR experiments/output/faster_rcnn/${NET}
set -x
set -e

NET=$1

#LOG="experiments/logs/classification/classification_${NET}_test.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
#exec &> >(tee -a "$LOG")
#echo Logging output to "$LOG"

time python2 tools/test_net.py \
    --cfg experiments/cfgs/e2e_classification_${NET}.yaml \
    TEST.WEIGHTS experiments/output/classification/${NET}/train/coco_disease_61class_train:coco_disease_61class_train_supplement/generalized_rcnn/model_iter11999.pkl \
    OUTPUT_DIR experiments/output/classification/${NET}
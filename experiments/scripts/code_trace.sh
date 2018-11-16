set -x
set -e

LOG="experiments/logs/code_trace.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"


# show function calls
python -m trace --trackcalls tools/train_net.py --cfg experiments/cfgs/e2e_mask_rcnn_resnet-50-FPN.yaml --multi-gpu-testing NUM_GPUS 1 USE_NCCL True OUTPUT_DIR tmp

# show all details
#python -m trace --trace tools/train_net.py --cfg experiments/cfgs/e2e_mask_rcnn_resnet-50-FPN.yaml --multi-gpu-testing NUM_GPUS 1 USE_NCCL True OUTPUT_DIR experiments/output/mask_rcnn

rm -rf tmp/

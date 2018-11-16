set -x
set -e

#GPU_NMS=$1
NET=$1

LOG="experiments/logs/rfcn_fpn/rfcn_end2end_${NET}-FPN.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time python2 tools/train_net.py \
    --cfg experiments/cfgs/e2e_rfcn_${NET}-FPN.yaml \
    OUTPUT_DIR experiments/output/rfcn_fpn/${NET}
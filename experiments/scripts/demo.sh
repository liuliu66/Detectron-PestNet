python2 tools/infer_simple.py \
    --cfg experiments/cfgs/e2e_mask_rcnn_resnet-50-FPN.yaml \
    --wts experiments/output/mask_rcnn_fpn/resnet-50/train/coco_chimeibing_single_trainval/generalized_rcnn/model_final.pkl \
    --output-dir demo/output \
    --image-ext jpg \
    demo
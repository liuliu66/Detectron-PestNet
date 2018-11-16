python2 tools/infer_cls.py \
    --cfg experiments/cfgs/e2e_classification_resnet-50-AT.yaml \
    --wts experiments/output/classification/resnet-50-AT/train/coco_disease_61class_train:coco_disease_61class_val:coco_disease_61class_train_supplement/generalized_rcnn/model_iter45999.pkl \
    --output-dir demo/output \
    --image-ext jpg \
    /data/data_disease_61class/testA
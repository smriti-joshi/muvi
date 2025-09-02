export nnUNet_preprocessed='/workspace/AutomaticSegmentation/nnUNet/nnunetv2/nnUNet_preprocessed'
export nnUNet_results='/workspace/AutomaticSegmentation/nnUNet/nnunetv2/nnUNet_results'
export nnUNet_raw='/workspace/AutomaticSegmentation/nnUNet/nnunetv2/nnUNet_raw'

python /workspace/muvi/scripts/infer_with_test_time_adaptation.py

python /workspace/muvi/scripts/infer_with_test_time_adaptation.py \
    --input /data/automatic-segmentation-nnunet-raw/nnUNet_raw/Dataset105_ISPY1Half/imagesTs \
    --output /workspace/AutomaticSegmentation/reproducibility/ispy_muvi_instancenorm \
    --model Dataset101_DukePhaseOneHalfMultifocal\
    --method muvi


python /workspace/muvi/scripts/compute_metrics.py\
    --pred_path /workspace/AutomaticSegmentation/reproducibility/ispy_muvi_instancenorm \
    --gt_path /workspace/AutomaticSegmentation/nnUNet/nnunetv2/data_external_validation/ISPY1/labelsTs_half
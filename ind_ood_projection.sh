OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python ind_ood_projection.py \
    --ind_dataset_dir data/ImageNet/train \
    --ood_dataset_dir ood_generated_samples/ \
    --save_dir outputs/projection_energy/ \
    --epochs 3 \
    --arch ViT-B/16 \
    --nshot 50 \
    --feat_dim 512
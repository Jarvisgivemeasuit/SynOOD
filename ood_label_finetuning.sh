OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python ood_label_finetuning.py \
    --ood_root ood_generated_samples \
    --model_name ViT-B/16 \
    --save_path outputs/ood_label_output_embeddings
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python ood_label_mining.py \
    --generate_ood_path ood_generated_samples/near_energy/ \
    --text_label_path texts/selected_neg_labels_in1k_10k.txt \
    --save_dir texts/neg_labels_for_gener_ood.txt \
    --emb_batch_size 1000
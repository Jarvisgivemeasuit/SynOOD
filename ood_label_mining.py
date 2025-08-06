import os
import clip
import torch
from datasets import get_generate_ood_loader
from tqdm import tqdm
import argparse

@torch.no_grad()
def compute_similarity_in_batches(text_features, image, model, device, batch_size=5000):
    text_features = text_features.to(torch.float32)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.to(torch.float32)
    
        sim = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    return sim


@torch.no_grad()
def ood_label_mining(generate_ood_path, text_label_path, save_dir, emb_batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device=device)
    dataloder = get_generate_ood_loader(generate_ood_path, transform=preprocess, batch_size=50, num_workers=8)

    texts = []
    with open(text_label_path, 'r') as file:
        for line in file:
            texts.append(line.strip())

    # get text tokens
    text_tokens = torch.cat([clip.tokenize(text) for text in texts]).to(device)
    text_features = []
    for i in range(0, len(text_tokens), emb_batch_size):
        text_features.append(model.encode_text(text_tokens[i:i + emb_batch_size]))
    text_features = torch.cat(text_features, dim=0)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    with open(save_dir, 'a') as f:
        for images, names in tqdm(dataloder):
            images = images.to(device)

            similarity = compute_similarity_in_batches(text_features, images, model, device)
            top_indices = similarity.argmax(dim=-1)

            top_labels = [texts[i] for i in top_indices]

            for name, label in zip(names, top_labels):
                f.write(f'{name}:{label}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OOD Label Mining with CLIP")
    parser.add_argument('--generate_ood_path', type=str, required=True, help='Path to generated OOD samples')
    parser.add_argument('--text_label_path', type=str, required=True, help='Path to text label file')
    parser.add_argument('--save_dir', type=str, required=True, help='Path to save mined labels')
    parser.add_argument('--emb_batch_size', type=int, default=1000, help='Batch size for text embedding')

    args = parser.parse_args()

    generate_ood_path = args.generate_ood_path
    text_label_path = args.text_label_path
    save_dir = args.save_dir
    emb_batch_size = args.emb_batch_size

    if not os.path.exists(save_dir):
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    ood_label_mining(generate_ood_path, text_label_path, save_dir, emb_batch_size)
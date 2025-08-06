import os
import torch
import clip
from tqdm import tqdm
from datasets import ImageNetForMultiClassification
from utils import AverageMeter
import argparse


def projection(cfg):
    clip_model, preprocess = clip.load(cfg.arch, device='cuda')
    dataset = ImageNetForMultiClassification(cfg.ind_dataset_dir, cfg.ood_dataset_dir, preprocess, cfg.nshot)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, 
                                             num_workers=32, pin_memory=True)
    print('Loaded dataset with {} samples.'.format(len(dataset)))

    classes = dataset.classes
    tokenized_prompts = torch.cat([clip.tokenize(l) for l in classes]).to('cuda')
    with torch.no_grad():
        text_features = clip_model.encode_text(tokenized_prompts).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    print(f'Loaded text features with shape: {text_features.shape}.')

    projection_layer = torch.nn.Sequential(
        torch.nn.Linear(cfg.feat_dim, cfg.feat_dim),
        torch.nn.LayerNorm(cfg.feat_dim),
        torch.nn.ReLU(),
        ).to('cuda')
    projection_layer.train()
    print('Loaded projection layer.')

    optim = torch.optim.Adam(projection_layer.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, cfg.epochs, 1e-5)
    print('Loaded optimizer.')

    losses = AverageMeter()
    for epoch in range(cfg.epochs):
        losses.reset()
        with tqdm(total=len(dataloader)) as t:
            for image, label in dataloader:
                image, label = image.to('cuda'), label.to('cuda')
                with torch.no_grad():
                    image_features = clip_model.encode_image(image).float()
                image_features = image_features + projection_layer(image_features)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                logit_scale = clip_model.logit_scale.exp()
                logits = logit_scale * image_features @ text_features.t()

                loss = torch.nn.functional.cross_entropy(logits, label)
                losses.update(loss.item(), image.size(0))
                
                optim.zero_grad()
                loss.backward()
                optim.step()

                t.set_postfix({'Epoch': epoch, 
                               'Loss': f'{losses.avg:.4f}', 
                               'LR': f'{optim.param_groups[0]["lr"]:.6f}'})
                t.update()
            sched.step()
        if not os.path.exists(cfg.save_dir):
            os.makedirs(cfg.save_dir)
        torch.save(projection_layer.state_dict(), f'{cfg.save_dir}/epoch_{epoch}.pt')

    t.close()
    print('Finished training.')


def get_config():
    parser = argparse.ArgumentParser(description='Projection Config')
    parser.add_argument('--ind_dataset_dir', type=str, default='data/ImageNet/train')
    parser.add_argument('--ood_dataset_dir', type=str, default='ood_generated_samples/')
    parser.add_argument('--save_dir', type=str, default='outputs/projection_energy/')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--arch', type=str, default='ViT-B/16')
    parser.add_argument('--nshot', type=int, default=50)
    parser.add_argument('--feat_dim', type=int, default=512)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    cfg = get_config()
    projection(cfg)
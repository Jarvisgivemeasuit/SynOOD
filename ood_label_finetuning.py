import os
import torch
import clip

from models.custom_clip import CustomCLIP
from datasets import OODLabelFinetuningDataset
from utils import ConstantWarmupScheduler, AverageMeter
from tqdm import tqdm
import argparse


def ood_embedding_finetuning(root, model_name, save_path, device='cuda'):
    # Load dataset
    clip_model, preprocess = clip.load(model_name, device=device)
    dataset = OODLabelFinetuningDataset(root, preprocess, device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, 
                                             num_workers=8, pin_memory=True)
    print('Loaded dataset with {} samples.'.format(len(dataset)))

    # Initialize the text embedding
    classes = dataset.classes
    tokenized_prompts = torch.cat([clip.tokenize(l) for l in classes]).to(device)
    with torch.no_grad():
        embeddings = clip_model.encode_text(tokenized_prompts)
    embeddings = torch.nn.Parameter(embeddings.type(clip_model.dtype), requires_grad=True)
    print(f'Initialized text embedding with shape:{embeddings.shape}.')

    # Load the model
    model = CustomCLIP(clip_model).to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    print('Loaded model.')

    # Load the optimizer
    epochs = 5
    warmup_epochs = 1
    warmup_cons_lr = 1e-5
    optim = torch.optim.SGD([{'params':embeddings}], lr=2e-2, momentum=0.9, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    # sched = ConstantWarmupScheduler(optim, sched, warmup_epochs, warmup_cons_lr)
    losses = AverageMeter()
    print('Loaded optimizer.')

    # Check if we need to use DataParallel
    device_count = torch.cuda.device_count()
    if device_count > 1:
        model = torch.nn.DataParallel(model)
    print('Using DataParallel with {} GPUs.'.format(device_count))

    # Train the embedding
    model.train()
    print('Start training.')
    for epoch in range(epochs):
        losses.reset()
        with tqdm(total=len(dataloader)) as t:
            for image, label in dataloader:
                image, label = image.to(device), label.to(device)

                logits = model.image_forward(image, embeddings)
                loss = torch.nn.functional.cross_entropy(logits, label)
                losses.update(loss.item(), image.size(0))

                optim.zero_grad()
                loss.backward()
                optim.step()

                t.set_postfix({'Epoch': epoch, 
                               'Loss': f'{losses.avg:.4f}', 
                               'LR': f'{optim.param_groups[0]["lr"]:.6f}'})
                t.update()

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(embeddings, f'{save_path}/epoch_{epoch}.pth')
        sched.step()
    t.close()
    print('Finished training.')


def inference(root, model_name, save_path, device='cuda'):
    # Load dataset
    clip_model, preprocess = clip.load(model_name, device=device)
    dataset = OODLabelFinetuningDataset(root, preprocess, device)
    classes = dataset.classes
    tokenized_prompts = torch.cat([clip.tokenize(l) for l in classes]).to(device)
    tokenized_prompts = tokenized_prompts.type(clip_model.dtype)

    # Load the model
    model = CustomCLIP(clip_model).to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    print('Loaded model.')

    # Load the text embedding
    embeddings = torch.load(os.path.join(save_path, 'epoch_0.pth'))

    # Check if we need to use DataParallel
    device_count = torch.cuda.device_count()
    if device_count > 1:
        model = torch.nn.DataParallel(model)
    print('Using DataParallel with {} GPUs.'.format(device_count))

    # Inference
    model.eval()
    print('Start inference.')
    with torch.no_grad():
        text_features = model.text_encoder(embeddings, tokenized_prompts)
        torch.save(text_features, os.path.join(save_path, 'text_features.pth'))


parser = argparse.ArgumentParser(description='OOD Label Finetuning')
parser.add_argument('--ood_root', type=str, default='ood_generated_samples', help='Dataset root path')
parser.add_argument('--model_name', type=str, default='ViT-B/16', help='CLIP model name')
parser.add_argument('--save_path', type=str, default='outputs/ood_label_output_embeddings', help='Path to save outputs')
args = parser.parse_args()

ood_root = args.ood_root
model_name = args.model_name
save_path = args.save_path

if __name__ == '__main__':
    ood_embedding_finetuning(ood_root, model_name, save_path)

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from collections import OrderedDict
from typing import List

import os
import random
import numpy as np
from tqdm import tqdm
from PIL import Image
import json


class ImageNetOneCategory(Dataset):
    def __init__(self, dataset_dir, category_id, transform, train=True) -> None:
        super().__init__()
        self.dataset_dir = os.path.join(dataset_dir, 'train') if train else os.path.join(dataset_dir, 'val')
        self.category_id = category_id
        self.transform   = transform

        self.categorys   = []
        with open('ind_name.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.categorys.append(line.strip().split(" ")[1])

        self.image_files = []
        self.category_dir = os.path.join(self.dataset_dir, self.categorys[self.category_id])
        for root, dirs, files in os.walk(self.category_dir):
            for file in files:
                if file.endswith('.JPEG'):
                    self.image_files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        image = self.transform(image)
        return image
    
    def get_image_path(self, idx):
        return self.image_files[idx]

    def get_category_name(self):
        return self.categorys[self.category_id]
    

class CleanedImageNet(Dataset):
    def __init__(self, dataset_dir, cleaned_dir, context_dir, height, width, transform):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.cleaned_dir = cleaned_dir
        self.context_dir = context_dir
        self.transform   = transform
        self.height      = height
        self.width       = width

        self.categorys = []
        with open('texts/ind_name.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.categorys.append(line.strip().split(" ")[1])
        
        with open(context_dir, 'r') as f:
            contexts = [json.loads(line) for line in f]

        self.contexts = {k: v for d in contexts for k, v in d.items()}
        self.cleaned_files = list(self.contexts.keys())

    def __len__(self):
        return len(self.cleaned_files)
    
    def __getitem__(self, idx):
        image  = Image.open(os.path.join(self.dataset_dir, 
                                         self.cleaned_files[idx])).convert('RGB').resize((self.height, self.width))
        mask   = Image.fromarray(np.ones((image.size[1], image.size[0])) * 255)
        prompt = self.contexts[self.cleaned_files[idx]][:73]
        prompt = f'A photo only remain {prompt}.'
        if self.transform:
            image = self.transform(image)
        cates, files = self.cleaned_files[idx].split('/')[1:]
        return image, mask, prompt, cates, files

    def collate_fn(self, batch):
        images, masks, prompts, cates, files = zip(*batch)
        return images, masks, prompts, cates, files
    

class ImageNetForBinary(Dataset):
    def __init__(self, ind_dataset_dir, ind_cleand_dir, ood_dataset_dir, ood_type:List, transform) -> None:
        super().__init__()
        self.ood_dataset_dir = ood_dataset_dir
        self.ind_cleand_dir  = ind_cleand_dir
        self.ood_type        = ood_type
        self.transform       = transform

        self.categorys = []
        with open('texts/ind_name.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.categorys.append(line.strip().split(" ")[1])

        with open(ind_cleand_dir, 'r') as f:
            contexts = [json.loads(line) for line in f]

        self.contexts = {k: v for d in contexts for k, v in d.items()}
        self.cleaned_files = list(self.contexts.keys())
        self.cleaned_files = [os.path.join(ind_dataset_dir, file) for file in self.cleaned_files]
        self.targets = [0] * len(self.cleaned_files)

        self.ood_image_files = []
        for type in os.listdir(ood_dataset_dir):
            if type not in ood_type:
                continue
            type_dir = os.path.join(ood_dataset_dir, type)
            if self.categorys[0] not in os.listdir(type_dir):
                self.ood_image_files += [os.path.join(type_dir, f) for f in os.listdir(type_dir) if f.endswith('.JPEG')]
            else:
                for cate in self.categorys:
                    category_dir = os.path.join(type_dir, cate)
                    for f in os.listdir(category_dir):
                        if f.endswith('.JPEG'):
                            self.ood_image_files.append(os.path.join(type_dir, cate, f))

        self.targets += [1] * len(self.ood_image_files)
        self.images = self.cleaned_files + self.ood_image_files

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.targets[idx]
    
    def get_image_path(self, idx):
        return self.image_files[idx]
   

class ImageNet(Dataset):
    def __init__(self, dataset_dir, transform, train=True, clean_dir=None) -> None:
        super().__init__()
        self.dataset_dir  = os.path.join(dataset_dir, 'train') if train else os.path.join(dataset_dir, 'val')
        self.transform    = transform
        self.classes_dict = {}
        with open('texts/ind_name.txt', 'r') as f:
                lines = f.readlines()
                for line in lines:
                    self.classes_dict[line.strip().split(" ")[1]] = f'a nice {line.strip().split(",")[1].strip()}'
        self.classes = list(self.classes_dict.values())

        if clean_dir:
            with open(clean_dir, 'r') as f:
                contexts = [json.loads(line) for line in f]

            self.contexts = {k: v for d in contexts for k, v in d.items()}
            self.cleaned_files = list(self.contexts.keys())
            self.labels = [self.classes.index(self.classes_dict[f.split('/')[0].strip()]) for f in self.cleaned_files]
            self.images = [os.path.join(dataset_dir, file) for file in self.cleaned_files]
        else:
            self.images = []
            self.labels = []
            for cate in self.classes_dict.keys():
                cate_dir = os.path.join(self.dataset_dir, cate)
                for f in os.listdir(cate_dir):
                    if f.endswith('.JPEG'):
                        self.images.append(os.path.join(cate_dir, f))
                        self.labels.append(self.classes.index(self.classes_dict[cate]))
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        image = self.transform(image)
        label = self.labels[idx]
        return image, label


num_ood_dict = {
    'iNaturalist': [10000, "iNaturalist"],
    'SUN':         [10000, "SUN"],
    'Places':      [10000, "Places"],
    'Textures':    [5640,  "dtd/images"],
}


class OODDataset(Dataset):
    def __init__(self, ood_root, dataset_name, transform, get_name=None) -> None:
        super().__init__()
        self.ood_root = os.path.join(ood_root, num_ood_dict[dataset_name][1])
        self.dataset_name = dataset_name
        self.transform = transform
        self.get_name = False if not get_name else get_name
        self.img_folders = os.listdir(self.ood_root)
        for i in self.img_folders:
            if i.startswith("."):
                self.img_folders.remove(i)
            if i.endswith(".txt"):
                self.img_folders.remove(i)
        self.imgs = []
        for i in self.img_folders:
            self.list = os.listdir(os.path.join(self.ood_root, i))
            self.imgs += [os.path.join(i, j) for j in self.list]
        for i in self.imgs:
            if not (i.endswith(".jpg") or i.endswith(".png") or i.endswith(".JPEG")):
                self.imgs.remove(i)

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        try:
            img = Image.open(os.path.join(self.ood_root, self.imgs[index])).convert("RGB")
        except:
            print(os.path.join(self.ood_root, self.imgs[index]))
        img = self.transform(img)
        if self.get_name:
            return img, self.imgs[index].split(".")[0]
        else:
            return img
        

class GenerateOODDataset(Dataset):
    def __init__(self, ood_root, transform, get_name=None) -> None:
        super().__init__()
        self.ood_root = ood_root
        self.transform = transform
        self.get_name = False if not get_name else get_name
        self.categorys = []
        with open('texts/ind_name.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                self.categorys.append(line.strip().split(" ")[1])

        self.imgs = []
        for i in self.categorys:
            self.list = os.listdir(os.path.join(self.ood_root, i))
            self.imgs += [os.path.join(i, j) for j in self.list]
        for i in self.imgs:
            if not (i.endswith(".jpg") or i.endswith(".png") or i.endswith(".JPEG")):
                self.imgs.remove(i)

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        try:
            img = Image.open(os.path.join(self.ood_root, self.imgs[index])).convert("RGB")
        except:
            print(os.path.join(self.ood_root, self.imgs[index]))
        img = self.transform(img)
        if self.get_name:
            return img, self.imgs[index].split(".")[0]
        else:
            return img


def get_generate_ood_loader(generate_ood_path, transform, batch_size, num_workers):
    def my_collate_fn(batch):
        imgs  = [i[0].unsqueeze(0) for i in batch]
        names = [i[1] for i in batch]
        imgs  = torch.vstack(imgs)
        return imgs, names
    dataset = GenerateOODDataset(generate_ood_path, transform=transform, get_name=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, 
                        pin_memory=True, drop_last=False, collate_fn=my_collate_fn)
    return loader


def get_ood_loaders(ood_root, transform, batch_size, num_workers):
    ood_loaders = {}
    def my_collate_fn(batch):
        imgs  = [i[0].unsqueeze(0) for i in batch]
        names = [i[1] for i in batch]
        imgs  = torch.vstack(imgs)
        return imgs, names

    for key in num_ood_dict:
        ood_data         = OODDataset(ood_root, key, transform=transform, 
                                      get_name=True
                                      )
        ood_loader       = DataLoader(ood_data, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=True, drop_last=False,
                                collate_fn=my_collate_fn
                                )
        ood_loaders[key] = ood_loader
    return ood_loaders


def get_neg_label(label_path):
    neg_labels = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            label = line.strip()
            neg_labels.append(label)
    return neg_labels


class OODLabelFinetuningDataset(torch.utils.data.Dataset):
    def __init__(self, root, preprocess, device='cuda'):
        self.device = device
        self.preprocess = preprocess

        self.ood_label_info = self.get_all_ood_info(5*10**4)
        self.cleaned_files = [
                              list(self.ood_label_info[0].keys()),
                              list(self.ood_label_info[1].keys()),
                              ]
        self.classes = list(set(
                        list(set(self.ood_label_info[0].values()))
                       +list(set(self.ood_label_info[1].values())),
                        ))

        self.images_path, self.labels = [], []
        for i in range(len(self.cleaned_files)):
            generate_folders = [
                f'near_energy_{i+1}', 
                f'near_msp_{i+1}', 
                f'near_no_grad_{i+1}',
                ]
            for folder in generate_folders:
                self.images_path += [os.path.join(root, folder, file) for file in self.cleaned_files[i]]
                self.labels += [self.classes.index(self.ood_label_info[i][file]) for file in self.cleaned_files[i]]

        generate_folders = ['far_neg_label']
        for folder in generate_folders:
            image_list = os.listdir(os.path.join(root, folder))
            self.images_path += [os.path.join(root, folder, file) for file in image_list]
            self.labels += [self.classes.index(' '.join(image.split('_')[:-2])) for image in image_list]

    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, idx):
        image = Image.open(self.images_path[idx]).convert('RGB')
        image_input = self.preprocess(image)
        label = self.labels[idx]
        return image_input, label
    
    def load_ood_info(self, ood_label_file):
        # Load the OOD labels
        with open(ood_label_file, 'r') as f:
            lines = f.readlines()
            ood_label_info = OrderedDict({f"{line.split(':')[0]}.JPEG": line.split(':')[1].strip() for line in lines})
        return ood_label_info
    
    def get_all_ood_info(self, num=5*10**4):
        ood_label_file = 'texts/neg_labels_for_gener_ood.txt'
        ood_label_info = self.load_ood_info(ood_label_file)

        return [ood_label_info]



class ImageNetForMultiClassification(Dataset):
    def __init__(self, ind_dataset_dir, ood_dataset_dir, transform) -> None:
        super().__init__()
        self.ood_dataset_dir = ood_dataset_dir
        self.transform       = transform
        self.ood_label_file  = 'texts/neg_labels_for_gener_ood.txt'

        # Load the OOD labels and file names
        self.ood_label_info = self.get_all_ood_info()
        self.files_names = [
                              list(self.ood_label_info[0].keys()),
                              list(self.ood_label_info[1].keys()),
                              ]
        self.ood_labels  = list(set(
                              list(set(self.ood_label_info[0].values()))
                             +list(set(self.ood_label_info[1].values())),
                            ))

        # Load the IND labels
        self.ind_labels, self.ind_dict = self.load_ind_labels()

        # Fuse the IND and OOD labels
        self.classes = self.ind_labels + self.ood_labels

        # List the IND images and labels
        self.ind_files = [os.path.join(ind_dataset_dir, file.replace("_ood", '')
                                       ) for file in self.files_names[0]]
        self.ind_class = [self.ind_dict[file.split('/')[0]] 
                                         for file in self.files_names[0]]
        self.ind_targets = [self.classes.index(c) for c in self.ind_class]

        # List the OOD images and labels
        self.ood_files, self.ood_targets = [], []
        generate_folders = [
                            'near_energy', 
                            # 'near_msp', 
                            # 'near_no_grad',
                            ]
        for folder in generate_folders:
            self.ood_files   += [os.path.join(ood_dataset_dir, folder, file) for file in self.files_names[0]]
            self.ood_targets += [self.classes.index(self.ood_label_info[0][file]) for file in self.files_names[0]]

        self.images = self.ind_files + self.ood_files
        self.targets = self.ind_targets + self.ood_targets


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        image = self.transform(image)
        return image, self.targets[idx]
    
    def get_image_path(self, idx):
        return self.image_files[idx]
    
    def load_ind_labels(self):
        # Load the IND labels
        ind_labels = []
        ind_dict = {}
        with open('texts/ind_name.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                label = line.split(",")[1].strip()
                label = f'a nice {label}'
                ind_labels.append(label)
                ind_dict[line.strip().split(" ")[1]] = label
        return ind_labels, ind_dict
    
    def load_ood_info(self, ood_label_file):
        # Load the OOD labels
        with open(ood_label_file, 'r') as f:
            lines = f.readlines()
            ood_label_info = OrderedDict({f"{line.split(':')[0]}.JPEG": line.split(':')[1].strip() for line in lines})
        return ood_label_info

    def get_all_ood_info(self):
        ood_label_file = 'texts/neg_labels_for_gener_ood.txt'
        ood_label_info = self.load_ood_info(ood_label_file)
        return [ood_label_info]

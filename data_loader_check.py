import os
import nltk
import torch
import torch.utils.data as data
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from torchvision import transforms
import pickle
from prepro import *
import random
def set_seed(seed_value=123):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# 设置种子，以确保每次数据加载顺序的一致性
set_seed()

class DataLoader(data.Dataset):
    def __init__(self, root, json, vocab, transform=None):

        self.root = root
        self.coco = COCO(json)
        self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        vocab = self.vocab
        ann_id = self.ids[index]
        captions1 = coco.anns[ann_id]['caption']
        # 4句中随机选1句进行训练
        rdn_index = np.random.choice(len(captions1), 1)[0]
        captions = captions1[rdn_index]
        path = coco.anns[ann_id]['image_id']
        
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
	
        tokens = nltk.tokenize.word_tokenize(str(captions).lower())
        caption1 = []
        caption1.append(vocab('<start>'))
        caption1.extend([vocab(token) for token in tokens])
        caption1.append(vocab('<end>'))
        target = torch.Tensor(caption1)
        return image, target, path

    def __len__(self):
        return len(self.ids)

def collate_fn(data):
    data.sort(key=lambda  x: len(x[1]), reverse=True)
    images, captions, paths = zip(*data)

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths, paths

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers):
    coco = DataLoader(root=root, json=json, vocab=vocab, transform=transform)

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
if __name__ == '__main__':
    data_transforms = {
        # data_transforms是一个字典，包含了两种数据变换（'train' 和 'valid'），分别用于训练和验证数据集。
        'train':
            transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        'valid':
            transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        # transforms.ToTensor() 和 transforms.Normalize() 用于将图像数据转换为张量并进行标准化。
    }
    with open("data/vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
    train_loader = get_loader("data/train2014", "data/train_caption.json", vocab,
                              data_transforms['train'], 1,
                              shuffle=True, num_workers=4)
    for i, (imgs, caps, caplens, path) in enumerate(train_loader):
        print(i)
        print(path)
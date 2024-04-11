import torch
import pickle
import argparse
import torch.nn.functional as F
from torchvision import transforms
from model import EncoderCNN, AttnDecoderRNN
from data_loader import get_loader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import numpy as np
from prepro import *
import matplotlib
from data_loader import get_loader
import torch
import numpy as np
import random

# 固定随机种子
def set_seed(seed_value=123):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# 设置种子，以确保每次数据加载顺序的一致性
set_seed()
matplotlib.use('TkAgg')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
from PIL import Image
from torchvision import transforms


def generate_caption(encoder, decoder, image, vocab, caps, caplens, max_length=20):
    caption_lengths = torch.tensor([max_length], dtype=torch.long).unsqueeze(0).to(device)
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    # image = Image.open(image_path).convert('RGB')
    # image = transform(image).unsqueeze(0).to(device)

    # 通过编码器获取图像特征
    encoder_out = encoder(image)
    # 初始化解码器的输入，假设vocab对象提供了'<start>'的索引
    start_token_idx = vocab('<start>')
    generated_idxs = [start_token_idx]  # 开始生成的索引列表
    inputs = torch.tensor([start_token_idx], dtype=torch.long).unsqueeze(0).unsqueeze(0).to(device)
    # caption_lengths = torch.tensor([max_length+1], dtype=torch.long).unsqueeze(0).to(device)

    # inputs = torch.tensor(start_token_idx).unsqueeze(0).unsqueeze(0).to(device)
    # print(inputs.shape)
    # 初始化存储生成的词索引
    generated_idxs = [start_token_idx]  # 开始生成的索引列表
    inputs = torch.tensor([start_token_idx], dtype=torch.long).unsqueeze(0).unsqueeze(0).to(device)


    outputs, _, _ = decoder(encoder_out, caps, caplens)
    # print(encoder_out.shape)
    for i in range(outputs.size(1)):
        # 通过解码器获取预测
        _, predicted_idx = torch.max(outputs[:, i, :], dim=1)  # 每个时间步选择最高得分
        generated_idxs.append(predicted_idx.item())
        if predicted_idx.item() == vocab('<end>'):
            break
        # 获取当前时间步最可能的单词索引
        # print(outputs.shape)
        # _, predicted_idx = torch.max(outputs, dim=2)  # 假设 outputs 形状是 [batch_size, seq_len, vocab_size]
        # current_word_idx = predicted_idx[:, -1]  # 取最后一个时间步的输出
        #
        # # 检查是否到达 '<end>'
        # if current_word_idx.item() == vocab('<end>'):
        #     break

        # generated_idxs.append(current_word_idx.item())
        # inputs = torch.cat((inputs, current_word_idx.unsqueeze(0).unsqueeze(0)), dim=1)  # 更新输入
        # _, predicted_idx = outputs.max(dim=2)
        # # 更新下一时间步的输入
        # inputs = predicted_idx
        # # 保存预测的词索引
        # idx = predicted_idx.item()
        # generated_idxs.append(idx)
        # # 检查是否达到'<end>'
        # if idx == vocab('<end>'):
        #     break
    # 将词索引转换为词
    generated_caption = [vocab.idx2word[idx] for idx in generated_idxs]
    # print(generated_caption)
    return ' '.join(generated_caption)
def generate_caption_simple(encoder, decoder, image_path, vocab, max_length=20):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    encoder_out = encoder(image)  # 编码图像
    start_token_idx = vocab('<start>')  # 获取开始标记
    inputs = torch.tensor([[start_token_idx]], dtype=torch.long).to(device)  # 初始化解码器输入

    generated_idxs = [start_token_idx]

    for _ in range(max_length):
        outputs, _, _ = decoder(encoder_out, inputs, torch.tensor([inputs.size(1)], dtype=torch.long).to(device))
        # print(outputs)
        _, predicted_idx = torch.max(outputs, dim=2)  # 获取预测的最可能的下一个单词索引
        # print(predicted_idx)
        predicted_idx = predicted_idx[:, -1]  # 选择最后一个时间步的输出
        if predicted_idx == vocab('<end>'):
            break  # 如果是结束标记则停止

        generated_idxs.append(predicted_idx.item())
        inputs = torch.cat((inputs, predicted_idx.unsqueeze(0).unsqueeze(0)), dim=1)  # 更新输入

    generated_caption = [vocab.idx2word[idx] for idx in generated_idxs if idx not in (vocab('<start>'), vocab('<end>'))]

    return ' '.join(generated_caption)


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
    train_loader = get_loader("data/val2014", "data/val_caption.json", vocab,
                              data_transforms['valid'], 1,
                              shuffle=True, num_workers=4)
    checkpoint = torch.load("checkpoint_mymodel-AB-LOSS.pth.tar", map_location=str(device))
    # checkpoint = torch.load("checkpoint_model_resnet101.pth.tar", map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()
    # image_path = 'chicken_image_516.jpg'
    # caption = generate_caption_simple(encoder, decoder, image_path, vocab)
    # print("caps", caps)
    # print("Generated Caption:", caption)
    # 索引转单词
    def indices_to_sentence(indices, vocab):
        # 确保每个索引都是整数
        return ' '.join(
            [vocab.idx2word[idx.item()] if isinstance(idx, torch.Tensor) else vocab.idx2word[idx] for idx in indices])

    for i, (imgs, caps, caplens) in enumerate(train_loader):
        caption =generate_caption(encoder, decoder, imgs, vocab, caps, caplens)
        # def generate_caption(encoder, decoder, image_path, vocab, caps, caplens, max_length=20):
        # 遍历caps中的每个序列并转换
        if i == 0:
            print("真实结果：")
            for cap in caps:
                sentence = indices_to_sentence(cap, vocab)
                print(sentence)
            print("预测结果：")
            print(caption)
            break
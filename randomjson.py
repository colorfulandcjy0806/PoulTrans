import json
import random
import os
import shutil

# 从captions.json文件中读取数据
with open('captions-test.json', 'r') as file:
    data = json.load(file)

# 直接使用整个数据集
annotations = data

# 打乱annotations的顺序
random.shuffle(annotations)

# 计算划分的索引
total_annotations = len(annotations)
print(total_annotations)
train_size = int(0.8 * total_annotations)
val_size = int(0.2 * total_annotations)

# 划分数据集

train_set = annotations[:train_size]
val_set = annotations[train_size:train_size + val_size]

# 创建存放训练集和验证集的文件夹
os.makedirs('data/train2014', exist_ok=True)
os.makedirs('data/val2014', exist_ok=True)

# 将图片移动到相应的文件夹
for item in train_set:
    image_path = f"images/{item['image_id']}"
    shutil.copy(image_path, 'data/train2014')

for item in val_set:
    image_path = f"images/{item['image_id']}"
    shutil.copy(image_path, 'data/val2014')

# 生成训练集的JSON文件
with open('data/train_caption.json', 'w') as train_file:
    json.dump(train_set, train_file, indent=2)

# 生成验证集的JSON文件
with open('data/val_caption.json', 'w') as val_file:
    json.dump(val_set, val_file, indent=2)

# 打印划分后的数据集大小
print("训练集大小:", len(train_set))
print("验证集大小:", len(val_set))

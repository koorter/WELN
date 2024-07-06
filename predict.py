import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import nn
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

import model_
from utils import get_yaml_value

# Configuration
classes = get_yaml_value("classes")
batchsize = get_yaml_value("batch_size")
data_dir = get_yaml_value("dataset_path")
image_size = get_yaml_value("image_size")
LPN = get_yaml_value("LPN")
net_path = "/media/sues/daa8aa38-6c2b-4fb6-a66f-327e4fc2f6a6/weights/University160K/eva02_L_1652.pth"
device = "cuda:0"

# Read name rank from file
name_rank = []
with open("query_drone_name.txt", "r") as f:
    name_rank = [line.strip() for line in f]

# Transformations
transform_test_list = [
    transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]

data_transforms = {
    'drone': transforms.Compose(transform_test_list),
    'satellite': transforms.Compose(transform_test_list)
}

# Custom dataset class
class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, file_list, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.samples = self._make_dataset(file_list)

    def _make_dataset(self, file_list):
        data = []
        for line in file_list:
            path = os.path.join(self.root, "query_drone_160k", line)
            data.append((path, 0))
        return data

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

# Data loaders
image_datasets = {
    'satellite': datasets.ImageFolder(
        os.path.join(data_dir, 'University160k', 'gallery'),
        data_transforms['satellite']
    ),
    'drone': CustomImageFolder(
        os.path.join(data_dir, 'University160k', 'university_160k_wx_test_set'),
        name_rank,
        data_transforms['drone']
    )
}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batchsize,
                                              shuffle=False, num_workers=32, pin_memory=True)
               for x in ['satellite', 'drone']}

# Ensure the drone dataset is sorted correctly
with open('query_drone_name.txt', 'r') as f:
    order = [line.strip() for line in f]
image_datasets['drone'].imgs = sorted(image_datasets['drone'].imgs, key=lambda x: order.index(x[0].split("/")[-1]))

# Load model
model = model_.eva02mim(701, 0.1).to(device)
model.load_state_dict(torch.load(net_path))

for i in range(4):
   cls_name = 'classifier' + str(i)
   c = getattr(model, cls_name)
   c.classifier = nn.Sequential()

model = model.eval()

# Utility functions
def rotate(img, k):
    return torch.rot90(img, k, [2, 3])

def fliplr(img):
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    return img.index_select(3, inv_idx)

def extract_feature(model, dataloader, block, LPN, view_index=1):
    features = torch.FloatTensor()
    for data in tqdm(dataloader):
        img, _ = data
        n, _, _, _ = img.size()

        if LPN:
            ff = torch.FloatTensor(n, 512, block).zero_().to(device)
        else:
            ff = torch.FloatTensor(n, 512).zero_().to(device)

        for i in range(2):
            if i == 1:
                img = fliplr(img)

            input_img = img.to(device)
            outputs = model(input_img, None) if view_index == 1 else model(None, input_img)
            ff += outputs

        if LPN:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(block)
            ff = ff.div(fnorm.expand_as(ff)).view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff.cpu()), 0)
    return features

# Extract features
query_feature = extract_feature(model, dataloaders["drone"], 4, LPN, 2)
gallery_feature = extract_feature(model, dataloaders["satellite"], 4, LPN, 1)

query_img_list = image_datasets["drone"].imgs
gallery_img_list = image_datasets["satellite"].imgs

# Matching results
result = {}
for i in range(len(query_img_list)):
    query = query_feature[i].view(-1, 1)
    score = torch.mm(gallery_feature, query).squeeze(1).cpu()
    index = torch.argsort(score, descending=True).numpy().tolist()
    max_score_list = index[:10]
    query_img = query_img_list[i][0]
    most_correlative_img = [gallery_img_list[idx][0] for idx in max_score_list]
    result[query_img] = most_correlative_img

# Save results
matching_table = pd.DataFrame(result)
matching_table.to_csv("result.csv")
print(matching_table)

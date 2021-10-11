from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import pandas as pd

from PIL import Image

# check if GPU device is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))

# define MTCNN
mtcnn = MTCNN(
    image_size=160,
    margin=14,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    device=device,
)


def collate_fn(x):
    return x[0]


# accessing names of people from folder names
dataset = datasets.ImageFolder("test")
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=0)

# 클래스 이름 확인
print(dataset.idx_to_class)

# define Inception Resnet V1 module
resnet = InceptionResnetV1(classify=False, num_classes=len(dataset.class_to_idx))
resnet.load_state_dict(torch.load("model_state_dict.pt"), strict=False)
resnet.eval().to(device)
# resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


# perform MTCNN face detection

# tensors
aligned = []
# names of class(사진 이름)
names = []

for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print(f"Face detected with probability: {prob}")
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])

# calculate image embeddings
aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()

# print distance matrix for classes
dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
print(pd.DataFrame(dists, columns=names, index=names))

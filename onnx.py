from models.inception_resnet_v1 import InceptionResnetV1

import torch
import torch.onnx
from torchvision import datasets
from torch.utils.data import DataLoader

# check if GPU device is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))

# define collate function
def collate_fn(x):
    return x[0]


# accessing names of people from folder names
dataset = datasets.ImageFolder("test_images")
idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=0)

# initializing resnet for face img to embeding conversion
resnet = InceptionResnetV1(classify=True, num_classes=len(dataset.class_to_idx))
resnet.load_state_dict(torch.load("model_state_dict.pt"), strict=False)
resnet.eval().to(device)

# convert pytorch to ONNX
dummy_data = torch.empty(1, 3, 224, 224, dtype=torch.float32)
torch.onnx.export(resnet, dummy_data, "output.onnx")

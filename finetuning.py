from models.mtcnn import MTCNN, fixed_image_standardization
from models.inception_resnet_v1 import InceptionResnetV1
from models.utils import training

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
import os

# path for image files
data_path = "images"

# set batch size and epochs
batch_size = 32
epochs = 10

# 0 if using cpu
workers = 0

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
    keep_all=False,
)

# face detection with MTCNN - iterate through the DataLoader and obtain cropped faces
dataset = datasets.ImageFolder(data_path, transform=transforms.Resize((512, 512)))

# replace face-cropped images from original images
dataset.samples = [
    (p, p.replace(data_path, data_path + "_cropped")) for p, _ in dataset.samples
]

# initiate DataLoader
loader = DataLoader(
    dataset, num_workers=workers, batch_size=batch_size, collate_fn=training.collate_pil
)

for i, (x, y) in enumerate(loader):
    mtcnn(x, save_path=y)
    print("\rBatch {} of {}".format(i + 1, len(loader)), end="")

# remove mtcnn to reduce GPU memory usage
del mtcnn

# define inception resnet v1 module
resnet = InceptionResnetV1(
    classify=True, pretrained="vggface2", num_classes=len(dataset.class_to_idx)
).to(device)

# define optimizer, scheduler, dataset, and dataloader
optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])

trans = transforms.Compose(
    [np.float32, transforms.ToTensor(), fixed_image_standardization]
)

dataset = datasets.ImageFolder(data_path + "_cropped", transform=trans)
img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)

# split train set and validation set using SubsetRandomSampler
train_inds = img_inds[: int(0.8 * len(img_inds))]
val_inds = img_inds[int(0.8 * len(img_inds)) :]

train_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds),
)

val_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds),
)

# define loss and evaluation functions
loss_fn = torch.nn.CrossEntropyLoss()
metrics = {"fps": training.BatchTimer(), "acc": training.accuracy}

# train model
writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

print("\n\nInitial")
print("-" * 10)

resnet.eval()
training.pass_epoch(
    resnet,
    loss_fn,
    val_loader,
    batch_metrics=metrics,
    show_running=True,
    device=device,
    writer=writer,
)

for epoch in range(epochs):
    print("\nEpoch {}/{}".format(epoch + 1, epochs))
    print("-" * 10)

    resnet.train()
    training.pass_epoch(
        resnet,
        loss_fn,
        train_loader,
        optimizer,
        scheduler,
        batch_metrics=metrics,
        show_running=True,
        device=device,
        writer=writer,
    )

    resnet.eval()
    training.pass_epoch(
        resnet,
        loss_fn,
        val_loader,
        batch_metrics=metrics,
        show_running=True,
        device=device,
        writer=writer,
    )

    # writer.add_scalar('Loss/train', , epoch)
    # writer.add_scalar('Loss/valid', , epoch)
    # writer.add_scalar('Accuracy/train', , epoch)
    # writer.add_scalar('Accuracy/valid', , epoch)

writer.close()

# 모델 객체의 state_dict 저장
torch.save(resnet.state_dict(), "model_state_dict.pt")

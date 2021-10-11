# importing libraries
from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image

import matplotlib.pyplot as plt
import cv2

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

# for multiple detection
mtcnn_all = MTCNN(
    image_size=160,
    margin=14,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    device=device,
    keep_all=True,
)

# define collate function
def collate_fn(x):
    return x[0]


# accessing names of people from folder names
dataset = datasets.ImageFolder("test_images")
idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=0)

# initializing resnet for face img to embeding conversion
resnet = InceptionResnetV1(classify=False, num_classes=len(dataset.class_to_idx))
resnet.load_state_dict(torch.load("model_state_dict.pt"), strict=False)
resnet.eval().to(device)

# loader = DataLoader(dataset, collate_fn=collate_fn)

# list of cropped faces from photos folder
face_list = []
# list of names corresponding to cropped photos
name_list = []
# list of embedding matrix after conversion from cropped faces to embedding matrix using resnet
embedding_list = []

for img, idx in loader:
    face, prob = mtcnn(img, return_prob=True)
    # if face detected with probability over 90%
    if face is not None and prob > 0.90:
        # passing cropped face into resnet model to get embedding matrix
        emb = resnet(face.unsqueeze(0))
        # resulten embedding matrix is stored in a list
        embedding_list.append(emb.detach())
        # names are stored in a list
        name_list.append(idx_to_class[idx])

# save data
data = [embedding_list, name_list]
torch.save(data, "data.pt")

# img_path= location of photo, data_path= location of *.pt file
def match_face(img_path, data_path):
    # loading model
    saved_data = torch.load("data.pt")
    # getting embedding data
    embedding_list = saved_data[0]
    # getting list of names
    name_list = saved_data[1]

    # getting embedding matrix of the given img
    img = Image.open(img_path)
    # returns cropped face and probability
    face, prob = mtcnn(img, return_prob=True)

    if face is not None:
        boxes, _ = mtcnn.detect(img)

        # detach is to make required gradient false
        emb = resnet(face.unsqueeze(0)).detach()

        # list of matched distances, minimum distance is used to identify the person
        dist_list = []

        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)

        min_dist = min(dist_list)
        min_dist_idx = dist_list.index(min_dist)

        name = name_list[min_dist_idx]

    else:
        print("Something went wrong!")

    return (name, min_dist, boxes)


# 테스트할 이미지 파일의 경로
IMAGE = "1.jpg"

result = match_face(IMAGE, "data.pt")

print("Face matched with: ", result[0], "With distance: ", result[1])

img = cv2.imread(IMAGE, cv2.IMREAD_COLOR)
plt.figure(figsize=(20, 10))
plt.imshow(img)

name = result[0]
min_dist = result[1]
box = result[2][0]

if min_dist < 0.56:
    cv2.putText(
        img,
        name + " " + str(round(min_dist, 4)),
        (int(box[0]), int(box[1])),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2
    )
else:
    cv2.putText(
        img,
        "UNKNOWN" + " " + str(round(min_dist, 4)),
        (int(box[0]), int(box[1])),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (0, 0, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2
    )


cv2.imshow("result", img)
# wait for 10 secs and destroy window
cv2.waitKey(10000)

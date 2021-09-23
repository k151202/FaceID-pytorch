import cv2
from PIL import Image

from models.mtcnn import MTCNN
from models.inception_resnet_v1 import InceptionResnetV1

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import pandas as pd

# check if GPU device is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: {}".format(device))

# define MTCNN
mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    device=device,
)

# for multiple detection
mtcnn_all = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    device=device,
    keep_all=True,
)

# define Inception Resnet V1 module

resnet = InceptionResnetV1(classify=False, num_classes=5)
resnet.load_state_dict(torch.load("model_state_dict.pt"), strict=False)
resnet.eval().to(device)

# define dataset and dataloader
def collate_fn(x):
    return x[0]


dataset = datasets.ImageFolder("images/")
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=0)

# list of names corresponding to cropped photos
name_list = []
# list of embedding matrix after conversion from cropped faces
# to embedding matrix using resnet
embedding_list = []

for img, idx in loader:
    face, prob = mtcnn(img, return_prob=True)
    if face is not None and prob > 0.92:
        print(f"Face detected with probability: {prob}")
        emb = resnet(face.unsqueeze(0))
        embedding_list.append(emb.detach())
        name_list.append(dataset.idx_to_class[idx])

# save data
data = [embedding_list, name_list]
torch.save(data, "data.pt")

# Using webcam to recognize face

load_data = torch.load("data.pt")
embedding_list = load_data[0]
name_list = load_data[1]

cap = cv2.VideoCapture(0)

while True:
    if not cap.isOpened():
        print("Failed to open webcam")
        break

    ret, frame = cap.read()
    if not ret:
        print("Failed to load frame")
        break

    img = Image.fromarray(frame)
    img_cropped_list, prob_list = mtcnn_all(img, return_prob=True)

    if img_cropped_list is not None:
        boxes, _ = mtcnn_all.detect(img)
        # print(boxes)
        # print(type(boxes[0]))

        for i, prob in enumerate(prob_list):
            if prob > 0.90:
                emb = resnet(img_cropped_list[i].unsqueeze(0)).detach()

                # list of matched distances
                dist_list = []

                for idx, emb_db in enumerate(embedding_list):
                    dist = torch.dist(emb, emb_db).item()
                    dist_list.append(dist)

                # get minimum distance value and index
                min_dist = min(dist_list)
                min_dist_idx = dist_list.index(min_dist)

                # get name corresponding to the minimum distance
                name = name_list[min_dist_idx]

                box = boxes[i]

                # storing copy of frame before drawing on it
                original_frame = frame.copy()

                if min_dist < 0.6:
                    frame = cv2.putText(
                        frame,
                        name + " " + str(round(min_dist, 4)),
                        (int(box[0]), int(box[1])),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    frame = cv2.putText(
                        frame,
                        "UNKNOWN" + " " + str(round(min_dist, 4)),
                        (int(box[0]), int(box[1])),
                        cv2.FONT_HERSHEY_DUPLEX,
                        1,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )

                frame = cv2.rectangle(
                    frame,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (0, 255, 0),
                    2,
                )

                cv2.imshow("IMG", frame)

                k = cv2.waitKey(1)

                if k == 27:
                    print("ESC pressed, closing...")
                    break

cap.release()
cv2.destroyAllWindows()

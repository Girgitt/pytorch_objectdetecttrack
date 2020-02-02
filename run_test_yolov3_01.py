from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.4
nms_thres=0.4

# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()
classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor
#Tensor = torch.FloatTensor


def prepare_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    return input_img


def detect_image(input_img):
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]

# load image and get detections
img_path = "images/blueangels.jpg"

img = Image.open(img_path)
exec_times = []
iters_done = 0
iters_max = 100
prepared_image = prepare_image(img)
for i in range(iters_max):
    iters_done += 1
    if iters_done % int(iters_max/10) == 0:
        print("done %s%%" % int(iters_done/int(iters_max/10)*10))
    prev_time = time.time()
    detections = detect_image(prepared_image)
    inference_time = datetime.timedelta(seconds=time.time() - prev_time)
    exec_times.append((inference_time))

average_timedelta = sum(exec_times, datetime.timedelta(0)) / len(exec_times)
print ('Inference Time: %s' % (average_timedelta))


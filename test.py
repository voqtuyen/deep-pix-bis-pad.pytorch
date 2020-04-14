import os
import cv2
import torch
from torchvision import transforms, datasets
from models.loss import PixWiseBCELoss
from models.densenet_161 import DeepPixBis
from datasets.PixWiseDataset import PixWiseDataset
from utils.utils import read_cfg, get_optimizer, build_network, get_device
from mtcnn import MTCNN
import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Setup epoch')
parser.add_argument('-e', '--epoch', help='Set up epoch number', default=35)
parser.add_argument('-c', '--conf', help='Set up confidence', type=float, default=0.5)

args = parser.parse_args()

cfg = read_cfg(cfg_file='config/patch-based_lr1e-3.yaml')

network = build_network(cfg)

epoch = args.epoch

checkpoint = torch.load(os.path.join(cfg['output_dir'], '{}_{}_{}.pth'.format(epoch, cfg['model']['base'], cfg['dataset']['name'])),  map_location=torch.device('cpu'))

network.load_state_dict(checkpoint['state_dict'])

network.eval()

capture = cv2.VideoCapture(0)
#capture = cv2.VideoCapture('http://192.168.4.104:8080/video')
#capture = cv2.VideoCapture('/home/tuyenvq/Videos/test1.mp4')
# face_detector = Detector(graph='models/frozen_inference_graph.pb')
detector = MTCNN()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((320,320)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

while capture.isOpened():
    ret, img = capture.read()
    
    if ret:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        boxes = detector.detect_faces(img_rgb)

        for box in boxes:
            box = box['box']
            box[0] = max(box[0], 0)
            box[1] = max(box[1], 0)

            hsv_crop = img_hsv[box[1]:(box[1] + box[3]), box[0]:(box[0] + box[2])]
            
            key2 = cv2.waitKey(1)
            if key2 == ord('w'):
                break

            hsv_crop = transform(hsv_crop)

            # print(anti_img.shape)
            hsv_crop = hsv_crop.unsqueeze(0)

            # print(anti_img.shape)

            dec = network.forward(hsv_crop)
            res = torch.mean(dec).item()
            print(res)
            cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255,0,0), 1)
            if res < args.conf:
                cv2.putText(img, 'Fake', (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
                cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0,0,255), 1)
            else:
                cv2.putText(img, 'Real', (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
                cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255,0,0), 1)
        cv2.imshow('Anti spoofing', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    else:
        break

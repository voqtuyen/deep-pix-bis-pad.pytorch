import os
import cv2
import torch
from torchvision import transforms, datasets
from models.loss import PixWiseBCELoss
from models.densenet_161 import DeepPixBis
from models.detector import Detector
from datasets.PixWiseDataset import PixWiseDataset
from utils.utils import read_cfg, get_optimizer, build_network, get_device
from mtcnn import MTCNN
import numpy as np
from PIL import Image


cfg = read_cfg(cfg_file='config/densenet_161_adam_lr1e-3.yaml')

network = build_network(cfg)

checkpoint = torch.load(os.path.join(cfg['output_dir'], '{}_{}.pth'.format(cfg['model']['base'], cfg['dataset']['name'])),  map_location=torch.device('cpu'))

network.load_state_dict(checkpoint['state_dict'])

network.eval()

capture = cv2.VideoCapture(0)
#capture = cv2.VideoCapture('http://192.168.4.104:8080/video')
# capture = cv2.VideoCapture('videos/Tuyen.mp4')
# face_detector = Detector(graph='models/frozen_inference_graph.pb')
detector = MTCNN()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

while capture.isOpened():
    ret, img = capture.read()
    
    if ret:
        img_det = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        boxes = detector.detect_faces(img_det)
        for box in boxes:
            box = box['box']
            box[0] = max(box[0], 0)
            box[1] = max(box[1], 0)

            anti_img = img_det[box[1]:(box[1] + box[3]), box[0]:(box[0] + box[2])]
            
            show_img = cv2.cvtColor(anti_img, cv2.COLOR_RGB2BGR)
            cv2.imshow('Test', show_img)
            key2 = cv2.waitKey(1)
            if key2 == ord('w'):
                break

            anti_img = transform(anti_img)

            # print(anti_img.shape)
            anti_img = anti_img.unsqueeze(0)

            # print(anti_img.shape)

            dec, binary = network.forward(anti_img)
            res = torch.mean(dec).item()
            print(res)
            cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255,0,0), 1)
            if res < 0.5:
                cv2.putText(img, 'Fake', (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
            else:
                cv2.putText(img, 'Real', (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
        cv2.imshow('Anti spoofing', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    else:
        break

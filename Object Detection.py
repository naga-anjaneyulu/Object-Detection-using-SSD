# Object Detection

# Importing the libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

# Detection function

def detect(frame,net,transform):
    # Transformations
    height,width = frame.shape[:2]
    frame_t = transform(frame)[0]
    x = torch.from_numpy(frame_t).permute(2,0,1)
    x = Variable(x.unsqueeze(0))
    
    # Feeding torch variable into pre trained SSD
    
    y = net(x)
    detections = y.data
    scale = torch.Tensor([width,height,width,height])
    
    for i in range(detections.size(1)):
        j = 0
        while detections[0,i,j,0] >= 0.6:
            pt = (detections[0,i,j,1:]*scale).numpy()
            cv2.rectangle(frame, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1
    return frame

# SSD Neural network


net = build_ssd('test')
#Pretrained SSD neural network 
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth',map_location = lambda storage ,loc: storage))


#Transformation

transform =  BaseTransform(net.size,(104/256.0,117/256.0,123/256.0))

# Object detection on a vedio

#reader = imageio.get_reader('Drone.mp4')
reader = imageio.get_reader('Horses.mp4')
fps = reader.get_meta_data()['fps']
#writer = imageio.get_writer('Drone_Output.mp4',fps = fps)
writer = imageio.get_writer('Horses_Output.mp4',fps = fps)
for i, frame in enumerate(reader):
    frame = detect(frame,net.eval(),transform)
    writer.append_data(frame)
    print(i)
writer.close()

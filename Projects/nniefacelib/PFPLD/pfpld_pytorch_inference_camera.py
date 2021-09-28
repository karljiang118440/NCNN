# created by Huang Lu
# 27/08/2016 17:05:45 
# Department of EE, Tsinghua Univ.

import cv2
import numpy as np




import argparse
import numpy as np
import torch
from torchvision import transforms
import cv2
import os
import glob
from pfld.pfld import PFLDInference
from hdface.hdface import hdface_detector
from pfld.utils import plot_pose_cube
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


det = hdface_detector(use_cuda=False)
checkpoint = torch.load("./models/pretrained/checkpoint_robust.pth")
plfd_backbone = PFLDInference().cuda()
plfd_backbone.load_state_dict(checkpoint)
plfd_backbone.eval()
plfd_backbone = plfd_backbone.cuda()
transform = transforms.Compose([transforms.ToTensor()])

cap = cv2.VideoCapture(0)
while(1):
    # get a frame
    ret, frame = cap.read()

    height, width = frame.shape[:2]

    img_det = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = det.detect_face(img_det)




    for i in range(len(result)):
        box = result[i]['box']
        cls = result[i]['cls']
        pts = result[i]['pts']
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 25))
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        size_w = int(max([w, h])*0.9)
        size_h = int(max([w, h]) * 0.9)
        cx = x1 + w//2
        cy = y1 + h//2
        x1 = cx - size_w//2
        x2 = x1 + size_w
        y1 = cy - int(size_h * 0.4)
        y2 = y1 + size_h

        left = 0
        top = 0
        bottom = 0
        right = 0
        if x1 < 0:
            left = -x1
        if y1 < 0:
            top = -y1
        if x2 >= width:
            right = x2 - width
        if y2 >= height:
            bottom = y2 - height

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(width, x2)
        y2 = min(height, y2)

        cropped = frame[y1:y2, x1:x2]
        print(top, bottom, left, right)
        cropped = cv2.copyMakeBorder(cropped, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
        
        cropped = cv2.resize(cropped, (112, 112))

        input = cv2.resize(cropped, (112, 112))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        input = transform(input).unsqueeze(0).cuda()
        pose, landmarks = plfd_backbone(input)
        poses = pose.cpu().detach().numpy()[0] * 180 / np.pi
        pre_landmark = landmarks[0]
        pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [size_w, size_h]
        cv2.rectangle(frame,(x1, y1), (x2, y2),(255,0,0))
        for (x, y) in pre_landmark.astype(np.int32):
            cv2.circle(frame, (x1 - left + x, y1 - bottom + y), 1, (255, 255, 0), 1)
        plot_pose_cube(frame, poses[0], poses[1], poses[2], tdx=pts['nose'][0], tdy=pts['nose'][1],
                    size=(x2 - x1) // 2)




    # show a frame
    cv2.imshow("capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows() 
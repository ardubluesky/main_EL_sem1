from ultralytics import YOLO
import cv2
model = YOLO('yolov8n.pt')
img_path = "./test.jpg"
results = model(img_path)
annotation_points = []
for r in results:
    for box in r.boxes:
        if int(box.cls[0])==0:
            x1,y1,x2,y2=box.xyxy[0]
            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)
            annotation_points.append([cx,cy])
            

from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter

img = Image.open(img_path).convert('RGB')
img_resized = img.resize((256,256))
img_array = np.array(img_resized,dtype=np.float32)/255.0

orig_img = cv2.imread(img_path)
orig_height, orig_width = orig_img.shape[:2]
scale_x = 256/orig_width
scale_y = 256/orig_height

gt = np.zeros((256,256),dtype=np.float32)
for pt in annotation_points:
    x,y = int(pt[0]*scale_x),int(pt[1]*scale_y)
    if 0<=y<256 and 0<=x<256 :
        gt[y,x]=1
density_map = gaussian_filter(gt,sigma=15)


import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(img_array)
plt.title('Preprocessed Image')
plt.scatter([x[0]*scale_x for x in annotation_points],[x[1]*scale_y for x in annotation_points],c='r',s=10)
plt.subplot(1,2,2)
plt.imshow(density_map,cmap='jet')
plt.title('Density Map')
plt.show()


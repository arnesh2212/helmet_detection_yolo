from ultralytics import YOLO

import tqdm
import numpy as np
#load model
model = YOLO("trainv4.pt")


image_path = 'demo.jpg'

#detect
results = model(image_path , show=True , save = True , conf=0.3)


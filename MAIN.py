from ultralytics import YOLO
import torch

model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# train on GPU 1
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


if __name__ == '__main__':

    model.train(data="C:\\Users\\arnes\\Desktop\\DATA\\HELMET_DATASET\\yolov8\\config.yaml", epochs=100,  device=device , save = True , save_period = 1 , batch = 32 )

    
    #save model to .pt file
    torch.save(model, "helmetv1.pt")
    model.export(format='onnx')





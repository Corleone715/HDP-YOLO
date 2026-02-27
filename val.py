import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os

if __name__ == '__main__':
    model = YOLO('runs/kzy_yolov8s-D2FPN-HFERB_backbone/Wise-Inner-PIoU/test/weights/best.pt') 
    model.val(data='/home/pci/data/kzy/ultralytics-main/dataset/data.yaml',
              split='test', 
              imgsz=640,
              batch=16,
              project='runs/kzy_yolov8s-D2FPN-HFERB_backbone/Wise-Inner-PIoU',
              name='exp',
              )
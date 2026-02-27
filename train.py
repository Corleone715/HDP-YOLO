import warnings, os
os.environ["CUDA_VISIBLE_DEVICES"]="1"    
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/kzy_yolov8s-D2FPN-HFE_backbone.yaml')
    model.train(data='C:/HDP-YOLO/dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=500,
                batch=32,
                close_mosaic=500,
                workers=4, 
                optimizer='SGD', 
                patience=100, # set 0 to close earlystop.
                project='runs/kzy_yolov8s-D2FPN-HFERB_backbone/Wise-Inner-PIoU',
                name='test',
                )

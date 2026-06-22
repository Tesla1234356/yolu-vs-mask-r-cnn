import os
import sys
import time
import cv2
import torch

# ==========================================
# IMPORTACIONES MASK R-CNN (DETECTRON2)
# ==========================================
sys.path.append(os.path.join(os.path.dirname(__file__), 'detectron2'))
try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog
    DETECTRON_INSTALLED = True
except ImportError as e:
    DETECTRON_INSTALLED = False
    print(f"Error cargando Detectron2: {e}")

# ==========================================
# IMPORTACIONES YOLOv8
# ==========================================
try:
    from ultralytics import YOLO
    YOLO_INSTALLED = True
except ImportError as e:
    YOLO_INSTALLED = False
    print(f"Error cargando Ultralytics: {e}")


class VisionManager:
    """Clase encargada de manejar toda la lógica y matemáticas de los modelos de Visión."""
    
    def __init__(self):
        self.yolo_model = None
        self.mask_rcnn_predictor = None
        self.mask_metadata = None
        self.has_gpu = torch.cuda.is_available()
        
    def check_hardware(self):
        gpu_name = torch.cuda.get_device_name(0) if self.has_gpu else "Ninguna (CPU)"
        print("="*50)
        print("DIAGNÓSTICO DE HARDWARE:")
        print(f"CUDA Disponible: {self.has_gpu}")
        print(f"Tarjeta Gráfica: {gpu_name}")
        print("="*50)
        return self.has_gpu, gpu_name

    def load_models(self):
        """Carga los pesos de YOLO y Mask R-CNN a la memoria."""
        self.check_hardware()
        
        # Cargar YOLO
        if YOLO_INSTALLED:
            self.yolo_model = YOLO("yolov8_5clases.pt")
            print(f"YOLO cargado con éxito. Ejecutando en: {'GPU' if self.has_gpu else 'CPU'}")
        else:
            raise Exception("Librería Ultralytics no instalada.")
            
        # Cargar Mask R-CNN
        if DETECTRON_INSTALLED:
            cfg = get_cfg()
            config_path = os.path.join(os.path.dirname(__file__), 'detectron2', 'configs', 'COCO-InstanceSegmentation', 'mask_rcnn_R_50_FPN_3x.yaml')
            cfg.merge_from_file(config_path)
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
            cfg.MODEL.WEIGHTS = "mask_rcnn_5clases.pth"
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
            
            if not self.has_gpu:
                cfg.MODEL.DEVICE = "cpu"
                
            self.mask_rcnn_predictor = DefaultPredictor(cfg)
            self.mask_metadata = MetadataCatalog.get("metadata_global")
            self.mask_metadata.thing_classes = ["botella_plastico", "lata", "vaso", "carton", "envoltura"]
            print("Mask R-CNN cargado con éxito.")
        else:
            raise Exception("Detectron2 no está instalado o configurado.")

    def predict_yolo(self, frame):
        """Infiere sobre una imagen usando YOLOv8. Retorna la imagen dibujada y el tiempo en ms."""
        if not self.yolo_model:
            return frame, 0
            
        t0 = time.time()
        # half=True acelera mucho en tarjetas NVIDIA como la GTX 1650
        results = self.yolo_model.predict(frame, conf=0.2, device=0 if self.has_gpu else 'cpu', verbose=False)
        annotated_frame = results[0].plot()
        elapsed_ms = (time.time() - t0) * 1000
        
        return annotated_frame, elapsed_ms

    def predict_mask_rcnn(self, frame):
        """Infiere sobre una imagen usando Mask R-CNN. Retorna la imagen dibujada y el tiempo en ms."""
        if not self.mask_rcnn_predictor:
            return frame, 0
            
        t0 = time.time()
        outputs = self.mask_rcnn_predictor(frame)
        v = Visualizer(frame[:, :, ::-1], metadata=self.mask_metadata, scale=1.0)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        annotated_frame = out.get_image()[:, :, ::-1]
        elapsed_ms = (time.time() - t0) * 1000
        
        return annotated_frame, elapsed_ms

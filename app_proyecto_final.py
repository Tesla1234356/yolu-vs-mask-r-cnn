import sys
import cv2
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QMessageBox, QTabWidget, QCheckBox, QSlider)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer

# Importamos la lógica de Visión Artificial que separamos en otro archivo
from vision_models import VisionManager

class ProyectoFinalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Proyecto Final: Detección de Objetos - RCNN vs YOLO")
        self.setGeometry(50, 50, 1400, 800)
        
        # ---------------------------------------------
        # 1. Variables de Lógica
        # ---------------------------------------------
        self.vision = VisionManager()
        self.current_cv_image = None
        
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_video_frame)
        
        # ---------------------------------------------
        # 2. Inicialización de GUI
        # ---------------------------------------------
        self.init_ui()
        self.load_vision_models()

    def init_ui(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QLabel { color: #ffffff; font-size: 14px; }
            QLabel#TitleLabel { font-size: 16px; font-weight: bold; background-color: #333; padding: 5px; border-radius: 4px; }
            QLabel#ImageLabel { border: 2px solid #555; background-color: #000; }
            QLabel#MetricsLabel { color: #ffeb3b; font-size: 15px; font-weight: bold; background-color: #222; padding: 5px; border-radius: 4px; }
            QPushButton { 
                background-color: #0d6efd; 
                color: white; 
                border-radius: 5px; 
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
            }
            QPushButton:hover { background-color: #0b5ed7; }
            QPushButton:disabled { background-color: #6c757d; }
            QPushButton#StopBtn { background-color: #dc3545; }
            QPushButton#StopBtn:hover { background-color: #bb2d3b; }
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #333; color: white; padding: 10px; font-size: 14px; font-weight: bold; }
            QTabBar::tab:selected { background: #0d6efd; }
        """)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        self.tabs = QTabWidget()
        self.tab_imagen = QWidget()
        self.tab_video = QWidget()
        
        self.tabs.addTab(self.tab_imagen, "📷 Análisis de Imágenes Estáticas")
        self.tabs.addTab(self.tab_video, "🎥 Análisis en Tiempo Real (Video/Webcam)")
        
        self.setup_tab_imagen()
        self.setup_tab_video()
        
        main_layout.addWidget(self.tabs)

    def load_vision_models(self):
        try:
            self.vision.load_models()
        except Exception as e:
            QMessageBox.warning(self, "Error al cargar modelos", str(e))

    # ==========================
    # LÓGICA DE PESTAÑA: IMAGEN
    # ==========================
    def setup_tab_imagen(self):
        layout = QVBoxLayout(self.tab_imagen)
        
        # Botones
        btn_layout = QHBoxLayout()
        self.btn_load_img = QPushButton("📸 Cargar Imagen Local")
        self.btn_load_img.clicked.connect(self.load_image)
        
        self.btn_predict_img = QPushButton("🔍 Ejecutar Ambos Modelos")
        self.btn_predict_img.clicked.connect(self.run_image_prediction)
        self.btn_predict_img.setEnabled(False)
        
        btn_layout.addWidget(self.btn_load_img)
        btn_layout.addWidget(self.btn_predict_img)
        layout.addLayout(btn_layout)

        # Controles de Umbral (Sliders)
        sliders_layout = QHBoxLayout()
        
        self.lbl_slider_yolo_img = QLabel("YOLO Umbral: 20%")
        self.lbl_slider_yolo_img.setStyleSheet("color: white; font-weight: bold;")
        self.slider_yolo_img = QSlider(Qt.Orientation.Horizontal)
        self.slider_yolo_img.setRange(5, 95)
        self.slider_yolo_img.setValue(20)
        self.slider_yolo_img.valueChanged.connect(self.update_img_labels)
        self.slider_yolo_img.sliderReleased.connect(self.run_image_prediction)
        
        self.lbl_slider_mask_img = QLabel("Mask R-CNN Umbral: 50%")
        self.lbl_slider_mask_img.setStyleSheet("color: white; font-weight: bold;")
        self.slider_mask_img = QSlider(Qt.Orientation.Horizontal)
        self.slider_mask_img.setRange(5, 95)
        self.slider_mask_img.setValue(50)
        self.slider_mask_img.valueChanged.connect(self.update_img_labels)
        self.slider_mask_img.sliderReleased.connect(self.run_image_prediction)
        
        sliders_layout.addWidget(self.lbl_slider_yolo_img)
        sliders_layout.addWidget(self.slider_yolo_img)
        sliders_layout.addWidget(self.lbl_slider_mask_img)
        sliders_layout.addWidget(self.slider_mask_img)
        layout.addLayout(sliders_layout)
        
        # Títulos
        titles_layout = QHBoxLayout()
        for t in ["Original", "YOLOv8 (Cajas)", "Mask R-CNN (Máscaras)"]:
            lbl = QLabel(t)
            lbl.setObjectName("TitleLabel")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            titles_layout.addWidget(lbl)
        layout.addLayout(titles_layout)

        # Imágenes
        img_layout = QHBoxLayout()
        self.lbl_img_orig = QLabel()
        self.lbl_img_yolo = QLabel()
        self.lbl_img_mask = QLabel()
        
        for lbl in [self.lbl_img_orig, self.lbl_img_yolo, self.lbl_img_mask]:
            lbl.setObjectName("ImageLabel")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_layout.addWidget(lbl)
            
        layout.addLayout(img_layout, stretch=1)
        
        # Tiempos
        time_layout = QHBoxLayout()
        self.lbl_time_img_yolo = QLabel("Tiempo YOLO: -- ms")
        self.lbl_time_img_mask = QLabel("Tiempo Mask R-CNN: -- ms")
        for lbl in [self.lbl_time_img_yolo, self.lbl_time_img_mask]:
            lbl.setObjectName("MetricsLabel")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            time_layout.addWidget(lbl)
        layout.addLayout(time_layout)

    def update_img_labels(self):
        self.lbl_slider_yolo_img.setText(f"YOLO Umbral: {self.slider_yolo_img.value()}%")
        self.lbl_slider_mask_img.setText(f"Mask R-CNN Umbral: {self.slider_mask_img.value()}%")

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Seleccionar Imagen", "", "Images (*.png *.jpg *.jpeg)")
        if file_name:
            self.current_cv_image = cv2.imread(file_name)
            if self.current_cv_image is not None:
                self.display_image(self.current_cv_image, self.lbl_img_orig)
                self.lbl_img_yolo.clear()
                self.lbl_img_mask.clear()
                self.lbl_time_img_yolo.setText("Tiempo YOLO: -- ms")
                self.lbl_time_img_mask.setText("Tiempo Mask R-CNN: -- ms")
                self.btn_predict_img.setEnabled(True)

    def run_image_prediction(self):
        if self.current_cv_image is None: return
        self.btn_predict_img.setText("Analizando...")
        self.btn_predict_img.setEnabled(False)
        QApplication.processEvents()
        
        yolo_thresh = self.slider_yolo_img.value() / 100.0
        mask_thresh = self.slider_mask_img.value() / 100.0
        
        # Ejecutar YOLO
        yolo_img, ms_yolo = self.vision.predict_yolo(self.current_cv_image, conf_thresh=yolo_thresh)
        self.display_image(yolo_img, self.lbl_img_yolo)
        self.lbl_time_img_yolo.setText(f"Tiempo YOLO: {ms_yolo:.1f} ms")
        QApplication.processEvents()

        # Ejecutar Mask R-CNN
        mask_img, ms_mask = self.vision.predict_mask_rcnn(self.current_cv_image, conf_thresh=mask_thresh)
        self.display_image(mask_img, self.lbl_img_mask)
        self.lbl_time_img_mask.setText(f"Tiempo Mask R-CNN: {ms_mask:.1f} ms")
                
        self.btn_predict_img.setText("🔍 Ejecutar Ambos Modelos")
        self.btn_predict_img.setEnabled(True)

    # ==========================
    # LÓGICA DE PESTAÑA: VIDEO
    # ==========================
    def setup_tab_video(self):
        layout = QVBoxLayout(self.tab_video)
        
        # Controles
        controls_layout = QHBoxLayout()
        self.btn_start_webcam = QPushButton("🟢 Iniciar Webcam")
        self.btn_start_webcam.clicked.connect(self.start_webcam)
        
        self.btn_load_video = QPushButton("📁 Cargar Archivo de Video")
        self.btn_load_video.clicked.connect(self.load_video)
        
        self.btn_stop_video = QPushButton("🛑 Detener")
        self.btn_stop_video.setObjectName("StopBtn")
        self.btn_stop_video.clicked.connect(self.stop_video)
        self.btn_stop_video.setEnabled(False)
        
        controls_layout.addWidget(self.btn_start_webcam)
        controls_layout.addWidget(self.btn_load_video)
        controls_layout.addWidget(self.btn_stop_video)
        layout.addLayout(controls_layout)
        
        # Checkboxes 
        checks_layout = QHBoxLayout()
        self.chk_yolo = QCheckBox("Activar YOLOv8")
        self.chk_yolo.setChecked(True)
        self.chk_yolo.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        
        self.chk_mask = QCheckBox("Activar Mask R-CNN")
        self.chk_mask.setChecked(False) # <--- Mejor lo dejamos apagado por defecto para probar YOLO
        self.chk_mask.setStyleSheet("color: white; font-size: 14px; font-weight: bold;")
        
        checks_layout.addWidget(self.chk_yolo)
        checks_layout.addWidget(self.chk_mask)
        checks_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(checks_layout)

        # Controles de Umbral para Video (Sliders)
        vid_sliders_layout = QHBoxLayout()
        
        self.lbl_slider_yolo_vid = QLabel("YOLO Umbral: 20%")
        self.lbl_slider_yolo_vid.setStyleSheet("color: white; font-weight: bold;")
        self.slider_yolo_vid = QSlider(Qt.Orientation.Horizontal)
        self.slider_yolo_vid.setRange(5, 95)
        self.slider_yolo_vid.setValue(20)
        self.slider_yolo_vid.valueChanged.connect(self.update_vid_labels)
        
        self.lbl_slider_mask_vid = QLabel("Mask R-CNN Umbral: 50%")
        self.lbl_slider_mask_vid.setStyleSheet("color: white; font-weight: bold;")
        self.slider_mask_vid = QSlider(Qt.Orientation.Horizontal)
        self.slider_mask_vid.setRange(5, 95)
        self.slider_mask_vid.setValue(50)
        self.slider_mask_vid.valueChanged.connect(self.update_vid_labels)
        
        vid_sliders_layout.addWidget(self.lbl_slider_yolo_vid)
        vid_sliders_layout.addWidget(self.slider_yolo_vid)
        vid_sliders_layout.addWidget(self.lbl_slider_mask_vid)
        vid_sliders_layout.addWidget(self.slider_mask_vid)
        layout.addLayout(vid_sliders_layout)
        
        # Títulos y Contenedores
        titles_layout = QHBoxLayout()
        for t in ["YOLOv8 (Tiempo Real)", "Mask R-CNN (Tiempo Real)"]:
            lbl = QLabel(t)
            lbl.setObjectName("TitleLabel")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            titles_layout.addWidget(lbl)
        layout.addLayout(titles_layout)

        vid_layout = QHBoxLayout()
        self.lbl_vid_yolo = QLabel("Esperando señal...")
        self.lbl_vid_mask = QLabel("Esperando señal...")
        
        for lbl in [self.lbl_vid_yolo, self.lbl_vid_mask]:
            lbl.setObjectName("ImageLabel")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFixedSize(600, 450)
            lbl.setScaledContents(True)
            vid_layout.addWidget(lbl)
            
        layout.addLayout(vid_layout, stretch=1)
        
        # FPS
        metrics_layout = QHBoxLayout()
        self.lbl_fps_yolo = QLabel("FPS YOLO: 0")
        self.lbl_fps_mask = QLabel("FPS Mask R-CNN: 0")
        for lbl in [self.lbl_fps_yolo, self.lbl_fps_mask]:
            lbl.setObjectName("MetricsLabel")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            metrics_layout.addWidget(lbl)
        layout.addLayout(metrics_layout)

    def update_vid_labels(self):
        self.lbl_slider_yolo_vid.setText(f"YOLO Umbral: {self.slider_yolo_vid.value()}%")
        self.lbl_slider_mask_vid.setText(f"Mask R-CNN Umbral: {self.slider_mask_vid.value()}%")

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.start_video_loop()

    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Seleccionar Video", "", "Videos (*.mp4 *.avi *.mkv)")
        if file_name:
            self.cap = cv2.VideoCapture(file_name)
            self.start_video_loop()

    def start_video_loop(self):
        if self.cap is not None and self.cap.isOpened():
            self.btn_start_webcam.setEnabled(False)
            self.btn_load_video.setEnabled(False)
            self.btn_stop_video.setEnabled(True)
            self.timer.start(1)

    def stop_video(self):
        self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_start_webcam.setEnabled(True)
        self.btn_load_video.setEnabled(True)
        self.btn_stop_video.setEnabled(False)
        self.lbl_vid_yolo.setText("Video Detenido")
        self.lbl_vid_mask.setText("Video Detenido")
        self.lbl_fps_yolo.setText("FPS YOLO: 0")
        self.lbl_fps_mask.setText("FPS Mask R-CNN: 0")

    def process_video_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.stop_video()
            return
            
        # Efecto espejo (voltear horizontalmente)
        frame = cv2.flip(frame, 1)
            
        frame = cv2.resize(frame, (600, 450))
        
        yolo_thresh = self.slider_yolo_vid.value() / 100.0
        mask_thresh = self.slider_mask_vid.value() / 100.0
        
        # Lógica delegada a la clase VisionManager
        if self.chk_yolo.isChecked():
            yolo_img, ms_yolo = self.vision.predict_yolo(frame, conf_thresh=yolo_thresh)
            fps_yolo = 1000.0 / (ms_yolo + 0.0001)
            self.display_fixed_image(yolo_img, self.lbl_vid_yolo)
            self.lbl_fps_yolo.setText(f"FPS YOLO: {fps_yolo:.1f}")
        else:
            self.lbl_vid_yolo.setText("YOLO Pausado")
            self.lbl_fps_yolo.setText("FPS YOLO: 0")
            
        if self.chk_mask.isChecked():
            mask_img, ms_mask = self.vision.predict_mask_rcnn(frame, conf_thresh=mask_thresh)
            fps_mask = 1000.0 / (ms_mask + 0.0001)
            self.display_fixed_image(mask_img, self.lbl_vid_mask)
            self.lbl_fps_mask.setText(f"FPS Mask R-CNN: {fps_mask:.1f}")
        else:
            self.lbl_vid_mask.setText("Mask R-CNN Pausado")
            self.lbl_fps_mask.setText("FPS Mask R-CNN: 0")

    # ==========================
    # UTILIDADES GUI
    # ==========================
    def display_image(self, img, labelWidget):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        labelWidget.setPixmap(pixmap.scaled(labelWidget.width(), labelWidget.height(), 
                                            Qt.AspectRatioMode.KeepAspectRatio, 
                                            Qt.TransformationMode.SmoothTransformation))

    def display_fixed_image(self, img, labelWidget):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qt_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img)
        labelWidget.setPixmap(pixmap)
        
    def closeEvent(self, event):
        self.stop_video()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProyectoFinalApp()
    window.show()
    sys.exit(app.exec())

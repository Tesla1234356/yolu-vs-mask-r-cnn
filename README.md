# Comparativa de Detección de Objetos: Mask R-CNN vs YOLOv8 ♻️

**Curso:** Visión Artificial - X ciclo, grupo B  
**Objetivo:** Comparar la detección de residuos sólidos (5 clases) usando un enfoque basado en regiones (Mask R-CNN) y un enfoque de una sola etapa (YOLOv8).

## 1. Arquitecturas Evaluadas
* **YOLOv8 Nano:** Arquitectura *Single-Stage* que prioriza la velocidad de inferencia a través de cajas delimitadoras (Bounding Boxes). Ideal para aplicaciones en tiempo real.
* **Mask R-CNN (ResNet50 FPN):** Arquitectura *Two-Stage* que recorta la silueta exacta del objeto (Segmentación de Instancias). Computacionalmente más pesada, pero extremadamente precisa delineando los bordes del objeto.

## 2. Métricas de Evaluación (Dataset Estandarizado COCO)
Tras entrenar ambos modelos con nuestro dataset estandarizado a 5 clases, se obtuvieron los siguientes resultados estáticos durante la evaluación sobre el conjunto de Validación. 

### Resumen General (mAP@50)
| Modelo | mAP@50 (Precisión General) | Precision (P) | Recall (R) |
| :--- | :---: | :---: | :---: |
| **Mask R-CNN** (Segmentación) | **56.72%** | N/A | 57.00% |
| **YOLOv8 Nano** (Cajas) | **37.40%** | 50.00% | 35.40% |

> *Nota:* Mask R-CNN superó en precisión a YOLOv8 Nano en esta prueba. Esto era de esperarse dado que Mask R-CNN utiliza un backbone masivo (`ResNet50`) diseñado para extraer características complejas, mientras que YOLOv8n es un modelo ultraligero diseñado para sacrificar un poco de precisión a cambio de máxima velocidad. Adicionalmente, YOLOv8 fue entrenado por un número reducido de épocas (50).

### Desglose de Precisión por Clases (mAP@50 o AP)
| Clase | Mask R-CNN (AP) | YOLOv8 (mAP50) |
| :--- | :---: | :---: |
| **Botella de Plástico** | 64.01% | 50.50% |
| **Lata** | 47.51% | **61.30%** |
| **Vaso** | 46.38% | 28.20% |
| **Envoltura** | 42.39% | 24.70% |
| **Cartón** | 30.40% | 22.40% |

## 3. Rendimiento en Tiempo Real (Video / Webcam)
Al ejecutar nuestra aplicación `app_proyecto_final.py` en hardware local (NVIDIA GTX 1650 con aceleración CUDA), encontramos comportamientos radicalmente distintos:

* **Velocidad (FPS):** YOLOv8 logra procesar el flujo de video a altísimas tasas de cuadros por segundo, permitiendo una experiencia completamente fluida. Mask R-CNN, debido a la complejidad de la red matemática que genera los polígonos de las máscaras, sufre un cuello de botella, reduciendo los FPS significativamente cuando se evalúa en el mismo entorno y en el mismo hilo de ejecución.
* **Calidad de Detección:** Mientras que YOLOv8 detecta el objeto rápidamente encerrándolo en un rectángulo, Mask R-CNN logra aislar los píxeles exactos de la basura. Esto sería indispensable si se usara, por ejemplo, un brazo robótico para recolectar el residuo sin tocar su entorno.

## 4. Conclusión
Si el objetivo es implementar una cámara de vigilancia en un basurero inteligente que cuente los residuos que caen **en tiempo real**, la aproximación basada en **YOLO es superior** por su inmensa velocidad. Si el objetivo es análisis en laboratorio o robótica de precisión donde se requiere aislar la forma geométrica del desecho, **Mask R-CNN es la opción adecuada** pese a su mayor costo computacional.

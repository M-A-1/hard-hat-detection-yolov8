# Hard Hat Detection Using YOLOv8

## Course Information

**Course:** AI / Machine Learning Course Project  
**Course Dates:** **23 August 2026 – 27 August 2026**  
**Project:** Hard Hat Detection Using YOLOv8  
**Student GitHub Repository:** https://github.com/M-A-1/hard-hat-detection-yolov8

## Project Overview

This project develops an object detection model for identifying hard-hat related objects in construction-worker images using YOLOv8.

The model was trained using the Hard Hat Workers dataset exported from Roboflow and implemented using the Ultralytics YOLO framework in Google Colab with an NVIDIA Tesla T4 GPU.

The model detects three object classes:
- `head`
- `helmet`
- `person`

The primary objective is to investigate the ability of a lightweight YOLOv8 model to detect heads and safety helmets in construction-related images.

## Project Objectives

1. Prepare a YOLO-compatible hard-hat detection dataset.
2. Train a YOLOv8 object detection model using transfer learning.
3. Evaluate the trained model using standard object-detection metrics.
4. Test the trained model on previously unseen images.
5. Save the trained model for subsequent inference.

## Dataset

The project uses the **Hard Hat Workers** dataset exported from Roboflow.

Dataset source:
https://universe.roboflow.com/shdv1/hard-hat-workers-aibtb/dataset/1

The dataset export contains **16,328 images** and uses YOLO-format annotations.

The dataset configuration defines three classes:

```text
0 - head
1 - helmet
2 - person
```

The dataset used for training and validation contained:
- Training: **14,209 images**
- Validation: **1,413 images**

The test images were used for qualitative inference after training.

## Model

**YOLOv8n (YOLOv8 Nano)** was used as a lightweight object-detection architecture. The model was initialized using pretrained YOLOv8n weights and fine-tuned on the Hard Hat Workers dataset.

### Environment
- Ultralytics YOLO
- PyTorch
- Python
- Google Colab
- NVIDIA Tesla T4 GPU

## Training Configuration

| Parameter | Value |
|---|---:|
| Model | YOLOv8n |
| Task | Object Detection |
| Epochs | 3 |
| Image size | 416 × 416 |
| Batch size | 16 |
| GPU | NVIDIA Tesla T4 |
| Training images | 14,209 |
| Validation images | 1,413 |
| Number of classes | 3 |

The training completed successfully, and the Ultralytics dataset scanner reported no corrupt images in the training or validation sets.

## Results

Final validation results:

| Metric | Overall Result |
|---|---:|
| Precision | **0.622** |
| Recall | **0.586** |
| mAP@50 | **0.632** |
| mAP@50–95 | **0.405** |

### Per-Class Performance

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| head | 0.922 | 0.870 | 0.925 | 0.598 |
| helmet | 0.944 | 0.887 | 0.952 | 0.610 |
| person | 0.000 | 0.000 | 0.019 | 0.008 |

The results show substantially stronger performance for the `head` and `helmet` classes than for the `person` class. The `person` class therefore represents an important limitation of the current mode[...]

## Training and Evaluation Outputs

The repository contains:
- `results.png` — training and validation metric/loss curves
- `results.csv` — recorded training results
- `confusion_matrix.png` — confusion matrix
- `confusion_matrix_normalized.png` — normalized confusion matrix
- `labels.jpg` — visualization of dataset labels

## Image Inference

After training, the model was tested on previously unseen test images. The model produced bounding-box predictions for objects including heads and helmets.

Example helmet detections had confidence scores of approximately 0.77, 0.76, 0.72, and 0.78.

These examples provide qualitative evidence that the trained model can identify helmet-related objects in images outside the training set. Individual confidence scores should not be interpreted as ove[...]

## Model Files

The main trained model is:
```text
best.pt
```

The dataset configuration is:
```text
data.yaml
```

## How to Use the Trained Model

Install Ultralytics:

```bash
pip install ultralytics
```

Load the model:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
```

Run inference on an image:

```python
results = model.predict(
    source="path/to/image.jpg",
    imgsz=416,
    conf=0.25,
    save=True
)
```

## Reproducing the Training

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="data.yaml",
    epochs=3,
    imgsz=416,
    batch=16
)
```

## Limitations

### 1. Short Training Duration
The final model was trained for only **3 epochs**. This demonstrates the training and inference pipeline but provides limited opportunity for extensive optimization.

### 2. Person-Class Performance
The `person` class showed very low validation performance:
- Precision: 0.000
- Recall: 0.000
- mAP@50: 0.019
- mAP@50–95: 0.008

Therefore, the current model should not be considered a reliable general-purpose person detector.

### 3. Class Distribution
The validation set contains substantially more `helmet` instances than `person` instances. The validation results show a large difference in performance between classes.

### 4. Limited Evaluation
The current project emphasizes image-based inference and quantitative validation. Further evaluation using a larger and more diverse independent dataset would be required before deployment in real con[...]

### 5. Safety-Critical Deployment
This model is a computer-vision research/prototype system. It should not be considered a replacement for workplace safety procedures, human supervision, or certified personal protective equipment comp[...]

## Future Work

- Train for more epochs.
- Perform hyperparameter optimization.
- Improve representation of underperforming classes.
- Review annotation quality and class distribution.
- Evaluate on additional independent images.
- Compare different YOLOv8 model sizes and image resolutions.
- Evaluate inference speed on target hardware.
- Extend the system to video-based detection and object tracking.
- Develop a real-time monitoring interface after adequate validation.

## Project Structure

```text
hard-hat-detection-yolov8/
│
├── README.md
├── best.pt
├── data.yaml
├── project_summary.txt
├── results.csv
├── results.png
├── confusion_matrix.png
├── confusion_matrix_normalized.png
└── labels.jpg
```

## Technologies

- Python
- YOLOv8
- Ultralytics
- PyTorch
- Google Colab
- NVIDIA CUDA
- Roboflow
- Computer Vision
- Object Detection

## SDAIA Reference

This project was completed as part of the course conducted from **23 August 2026 to 27 August 2026**.

The official **SDAIA GitHub organization** is referenced as part of the project's course/technical context:

https://github.com/SDAIA

SDAIA is the Saudi Data & AI Authority and is the Kingdom's national reference for data and artificial intelligence. The official SDAIA website is:

https://sdaia.gov.sa/

The SDAIA GitHub organization is an external reference and is not presented as the source of the Hard Hat Workers dataset or as the author of this project.

## Dataset Reference

Hard Hat Workers dataset, Roboflow Universe:

https://universe.roboflow.com/shdv1/hard-hat-workers-aibtb/dataset/1

## Model Reference

Ultralytics YOLO:

https://github.com/ultralytics/ultralytics

## Conclusion

A YOLOv8n object-detection model was successfully trained on the Hard Hat Workers dataset to detect `head`, `helmet`, and `person` classes.

The final model achieved an overall **mAP@50 of 0.632** and **mAP@50–95 of 0.405** on the validation set.

Performance was particularly strong for the `head` and `helmet` classes, with mAP@50 values of **0.925** and **0.952**, respectively. In contrast, the `person` class performed poorly and represents a [...]

The project demonstrates a functioning YOLOv8 hard-hat detection pipeline while documenting the limitations that should be addressed in future development.

## References

1. **SDAIA GitHub:** https://github.com/SDAIA
2. **SDAIA official website:** https://sdaia.gov.sa/
3. **Hard Hat Workers dataset (Roboflow):** https://universe.roboflow.com/shdv1/hard-hat-workers-aibtb/dataset/1
4. **Ultralytics YOLO:** https://github.com/ultralytics/ultralytics

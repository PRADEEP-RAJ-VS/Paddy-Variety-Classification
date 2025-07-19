---

# 🌾 Paddy Leaf Variety Classification using MobileNetV2 + Custom Filters

This project presents a robust image classification pipeline for identifying **paddy Variety classification** using **MobileNetV2** and advanced **custom image preprocessing**. It includes noise reduction, contrast enhancement, and Gabor filtering, ensuring optimal model performance even on noisy agricultural image datasets.

---

## 📌 Objective

To accurately classify paddy leaf images into their respective Variety using:

* A **custom hybrid preprocessing pipeline**
* A **fine-tuned MobileNetV2 model**
* Performance metrics (accuracy, precision, recall) and visualizations

---

## 🛠️ Features

* ✅ Custom Image Enhancement using OpenCV:

  * Bilateral Filtering
  * CLAHE (Contrast Limited Adaptive Histogram Equalization)
  * Gabor Filtering
* ✅ Image Augmentation with `ImageDataGenerator`
* ✅ Fine-tuning MobileNetV2 on custom dataset
* ✅ EarlyStopping + ModelCheckpoint callbacks
* ✅ Evaluation using Confusion Matrix & Classification Report
* ✅ Predicts classes on unseen test images

---

## 🗂️ Folder Structure

```
paddy_dataset/
├── train/
├── validation/
└── test/
```

Each subfolder should contain one folder per class (Variety type), with corresponding images inside.

---

## 📦 Requirements

Install these before running:

```bash
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn
```

---

## 🧪 Image Preprocessing

A custom OpenCV pipeline enhances image quality before feeding to the model.

```python
def apply_custom_filter(image_path):
    img = cv2.imread(image_path.decode())
    img = cv2.resize(img, (224, 224))
    bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    laplacian = cv2.Laplacian(bilateral, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    lab = cv2.cvtColor(laplacian, cv2.COLOR_BGR2LAB)
    cl = cv2.createCLAHE(clipLimit=2.0).apply(lab[..., 0])
    lab_enhanced = cv2.merge((cl, lab[..., 1], lab[..., 2]))
    enhanced_lab = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # Gabor Filter
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gabor = np.max([
        cv2.filter2D(gray, cv2.CV_8UC3, cv2.getGaborKernel((21, 21), 4.0, theta, 10.0, 0.5))
        for theta in np.arange(0, np.pi, np.pi / 4)
    ], axis=0)
    gabor_color = cv2.merge([gabor]*3)

    hybrid = cv2.addWeighted(enhanced_lab, 0.6, gabor_color, 0.4, 0)
    return hybrid.astype(np.uint8)
```

---

## 🧠 Model Architecture

```python
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
```

* **Base**: MobileNetV2 (pretrained on ImageNet)
* **Trainable Layers**: Last N layers (`fine_tune_at=100`)
* **Loss**: Sparse categorical crossentropy
* **Optimizer**: RMSprop (lr = 1e-5)

---

## 🔁 Training Details

```python
model.fit(train_dataset,
          epochs=10,
          validation_data=validation_dataset,
          callbacks=[EarlyStopping(patience=2, restore_best_weights=True),
                     ModelCheckpoint("best_finetuned_model.h5", save_best_only=True)])
```

---

## 📊 Evaluation

### Accuracy Graph:

```python
plt.plot(history.history['accuracy'], label="Training")
plt.plot(history.history['val_accuracy'], label="Validation")
```

### Confusion Matrix (Normalized):

```python
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens',
            xticklabels=class_labels, yticklabels=class_labels)
```

### Classification Report:

```text
Precision, Recall, F1-score for each class
```

---

## 📷 Prediction on Test Images

```python
for file in os.listdir('paddy_dataset/test'):
    img = image.load_img(file_path, target_size=(200, 200))
    prediction = model.predict(preprocessed_img)
    print(f"Predicted Class: {class_labels[np.argmax(prediction)]}")
```

---

## 📌 Results

* 📈 Achieved high validation accuracy with minimal overfitting
* 🧪 Strong performance across multiple Varity categories
* 🔍 Model interpretable via confusion matrix and class reports

---

## 🚀 Future Work

* Deploy model using Flask or Streamlit
* Add real-time webcam integration for live classification
* Explore other architectures (EfficientNet, ResNet)
* Introduce Grad-CAM for visual explanations

---

## 🧑‍💻 Author

**Pradeep Raj V S** – Final Year B.Tech IT
This is part of my ML-based project focusing on agriculture + deep learning.

---

## 📜 License

This project is open source under the [MIT License](https://opensource.org/licenses/MIT).

---

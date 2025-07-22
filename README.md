# 💊 Drug and Vitamin Classification using Deep Learning

This project involves building a deep learning model using transfer learning to classify pharmaceutical drugs and vitamins based on their images.

## 📂 Dataset

The dataset used is from Kaggle:  
🔗 [Pharmaceutical Drugs and Vitamins Synthetic Images](https://www.kaggle.com/datasets/vencerlanz09/pharmaceutical-drugs-and-vitamins-synthetic-images)  
It includes synthetic images of different drugs and vitamins, categorized into 10 different classes.

## 🧠 Model Architecture

The model utilizes **MobileNetV2** (pretrained on ImageNet) as the base for transfer learning. Additional dense layers are added for classification.

### ✅ Key Components:
- **Preprocessing**: Resizing to 224x224 and MobileNetV2 preprocessing.
- **Data Augmentation**: `ImageDataGenerator` with validation split.
- **Model Head**:
  - Dense(256, ReLU) → Dropout
  - Dense(256, ReLU) → Dropout
  - Dense(10, Softmax)
- **Callbacks**:
  - `EarlyStopping`
  - `ModelCheckpoint` (best val_accuracy)

## 🛠️ Technologies & Libraries
- Python
- TensorFlow / Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

## 📈 Training Performance

Model was trained for **8 epochs**, with early stopping to prevent overfitting.

### Visualizations:
- Training vs Validation Accuracy  
- Training vs Validation Loss  
- Predictions with color-coded correctness (green = correct, red = wrong)

## 📊 Evaluation

- Model evaluated using test set.
- Metrics reported using `classification_report` from `sklearn`.

### Example output:
```
              precision    recall  f1-score   support
   VitaminC       0.92      0.94      0.93       100
   DrugX         0.88      0.85      0.86       100
   ...
   accuracy                          0.90       1000
```

## 📌 Folder Structure
```
Drug_and_Vitamin_Classification_using_DL/
│
├── Drug Vision/
│   └── Data Combined/         # Image folders by class
├── model_training.py          # All model training code
├── README.md
```

## 📥 How to Run

1. Clone the repository.
2. Download the dataset from Kaggle and place it under `Drug Vision/Data Combined`.
3. Run the Python script (`model_training.py`).
4. Output: Evaluation scores + graphs.

## ⚠️ Note

- Ensure you have the required libraries installed (`tensorflow`, `sklearn`, `matplotlib`, etc.).
- This project uses **MobileNetV2**, which requires internet access for initial weights download.

## 📄 License

MIT License

# 🖼️ PyTorch Image Classifier  

A simple deep learning project for **binary image classification** using **PyTorch**.  
This project trains a Convolutional Neural Network (CNN) and allows you to classify new images using a Tkinter file dialog.  

---

## 🚀 Features  
- Train a CNN on a custom dataset with two classes  
- Apply data augmentation (random crops, flips, normalization)  
- Save and reload trained models (`.pth` file)
- Classify new images with a GUI file picker  
- Achieves ~92% accuracy on sample datasets  

---

## 🛠️ Technologies  
- Python 3.x  
- PyTorch  
- Torchvision  
- Pillow (PIL)  
- Tkinter (for GUI)  

---

## 📂 Files in This Repo  
- **`train.py`** → trains the CNN on your dataset and saves `image_classifier.pth`  
- **`imageClassifier.py`** → loads the trained model and classifies a selected image  

---

## 📊 Training  

1. Place your dataset in a folder with **two subfolders** (one per class). Example:  
dataset/
├── class_1/
│ ├── img1.jpg
│ ├── img2.jpg
├── class_2/
├── img3.jpg
├── img4.jpg


2. Run the training script:  
```bash
python train.py
```
3. Run the classifier:
```bash
python imageClassifier.py
```


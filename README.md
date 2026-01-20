# AIML-TASK-4
# Adult Income Dataset – Data Preprocessing

## 📌 Project Overview
This project focuses on preprocessing the **Adult Income Dataset** to make it suitable for Machine Learning models.  
The dataset is cleaned, encoded, scaled, and saved in a fully **ML-ready format**.

The goal is to predict whether a person earns **more than $50K per year** based on demographic and employment-related features.

---

## 📂 Dataset Information
- **Dataset Name:** Adult Income Dataset
- **Problem Type:** Supervised Learning (Binary Classification)
- **Target Variable:** `income`
  - `0` → ≤50K
  - `1` → >50K
- **Original Format:** Mixed (categorical + numerical)
- **Final Format:** Fully numerical (model-ready)

---

## 🛠️ Preprocessing Steps Performed

### 1️⃣ Data Loading
- Loaded dataset using **Pandas**
- Inspected structure, datatypes, and null values

---

### 2️⃣ Missing Value Handling
- Replaced `' ?'` with `NaN`
- Removed rows containing missing values to ensure data quality

---

### 3️⃣ Feature Type Identification
**Numerical Features**
- age  
- fnlwgt  
- education-num  
- capital-gain  
- capital-loss  
- hours-per-week  

**Categorical Features**
- workclass  
- education  
- marital-status  
- occupation  
- relationship  
- race  
- sex  
- native-country  

---

### 4️⃣ Encoding Strategy

#### 🔹 Label Encoding (Ordered Feature)
- **education** encoded using ordinal ranking  
  (Preschool → Doctorate)

#### 🔹 Target Encoding
- `income` mapped to binary values:
  - `<=50K` → 0
  - `>50K` → 1

#### 🔹 One-Hot Encoding
- Applied to all nominal categorical features
- `drop_first=True` used to avoid dummy variable trap

---

### 5️⃣ Feature Scaling
- Applied **StandardScaler**
- Numerical features transformed to:
  - Mean ≈ 0
  - Standard Deviation ≈ 1

---

## 📈 Why Scaling Matters
Scaling improves performance for:
- Logistic Regression
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- Neural Networks

> Note: Tree-based models like Decision Trees and Random Forests do not require scaling.

---

## 💾 Output File
- **Preprocessed Dataset:** `adult_preprocessed.csv`
- Fully numerical
- Ready for model training and evaluation

---

## 🚀 How to Use
```python
import pandas as pd

df = pd.read_csv("adult_preprocessed.csv")

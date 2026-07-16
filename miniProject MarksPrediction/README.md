# 📈 Student Marks Prediction using Linear Regression

> A beginner-friendly Machine Learning mini project that predicts student marks based on study hours using **Simple Linear Regression** in Python.

![Python](https://img.shields.io/badge/Language-Python-blue?style=flat-square)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Linear%20Regression-green?style=flat-square)
![Library](https://img.shields.io/badge/Library-scikit--learn-orange?style=flat-square)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=flat-square)

---

## 📌 Project Overview

This project demonstrates the implementation of **Simple Linear Regression** using the **scikit-learn** library.

A synthetic dataset of student study hours and corresponding marks is generated. The model is trained to learn the relationship between study hours and marks, evaluated using standard regression metrics, and then used to predict marks for new study-hour values.

---

## ✨ Features

- 📊 Synthetic dataset generation
- 🧹 Data preprocessing using Pandas
- 🔀 Train-Test Split
- 🤖 Linear Regression model training
- 📈 Model evaluation using R² Score and MAE
- 🔮 Prediction for new study-hour values
- 📉 Regression line visualization
- 💾 Dataset and plot saved automatically

---

## 🛠️ Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

## 📂 Project Structure

```
miniProject/
│── marks_predictor.py
│── student_marks_dataset.csv
│── marks_prediction_plot.png
│── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/amanverma123-prog/XtraGrad-Internship.git
```

Navigate to the project folder

```bash
cd XtraGrad-Internship/miniProject
```

Install dependencies

```bash
pip install numpy pandas matplotlib scikit-learn
```

---

## ▶️ Run the Project

```bash
python marks_predictor.py
```

---

## 📊 Sample Output

```
Learned equation:
Marks = 6.42 + 8.31 × StudyHours

Model Evaluation

R² Score : 0.91
MAE      : 4.83

Predictions

Study Hours: 2.0  → 22.9 Marks
Study Hours: 5.0  → 48.1 Marks
Study Hours: 7.5  → 69.2 Marks
```

*(Values may vary because the dataset is generated randomly.)*

---

## 📈 Model Evaluation Metrics

- **R² Score** measures how well the regression model fits the data.
- **Mean Absolute Error (MAE)** measures the average prediction error.

---

## 📷 Output

The project generates:

- **student_marks_dataset.csv** → Generated dataset
- **marks_prediction_plot.png** → Scatter plot with regression line

---

## 🎯 Learning Outcomes

- Understanding Simple Linear Regression
- Creating datasets using NumPy
- Data manipulation with Pandas
- Training ML models using Scikit-learn
- Evaluating regression models
- Visualizing predictions using Matplotlib

---

## 🚀 Future Improvements

- Use a real student performance dataset
- Add multiple input features
- Build a GUI using Tkinter
- Deploy using Streamlit or Flask
- Compare multiple regression algorithms

---

## 👨‍💻 Author

**Aman Verma**

- GitHub: https://github.com/amanverma123-prog

---

## ⭐ If you found this project useful, consider giving the repository a star!

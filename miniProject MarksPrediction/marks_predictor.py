import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# 1. Generate synthetic dataset
np.random.seed(42)

n_samples = 60
study_hours = np.round(np.random.uniform(0.5, 10, n_samples), 1)

noise = np.random.normal(0, 6, n_samples)
marks = 5 + 8.5 * study_hours + noise
marks = np.clip(marks, 0, 100)

df = pd.DataFrame({
    "StudyHours": study_hours,
    "Marks": np.round(marks, 1)
})

print("Sample of dataset:")
print(df.head(), "\n")
print(f"Dataset size: {len(df)} rows\n")

df.to_csv("student_marks_dataset.csv", index=False)

# 2. Train/test split
X = df[["StudyHours"]]  # feature matrix (2D)
y = df["Marks"]          # target vector

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

print(f"Learned equation: Marks = {model.intercept_:.2f} + {model.coef_[0]:.2f} * StudyHours\n")

# 4. Evaluate on test data
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("Model Evaluation on Test Set:")
print(f"  R^2 score : {r2:.3f}  (1.0 = perfect fit, closer to 1 is better)")
print(f"  MAE       : {mae:.2f} marks (average prediction error)\n")

# 5. Predict marks for new input
sample_hours = pd.DataFrame({"StudyHours": [2, 5, 7.5]})
sample_preds = model.predict(sample_hours)

print("Predictions for new study-hour values:")
for hours, pred in zip(sample_hours["StudyHours"], sample_preds):
    print(f"  Study Hours: {hours} -> Predicted Marks: {pred:.2f}")

# 6. Visualize
plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, color="steelblue", label="Training data")
plt.scatter(X_test, y_test, color="orange", label="Test data")

x_line = pd.DataFrame({"StudyHours": np.linspace(0, 10, 100)})
y_line = model.predict(x_line)
plt.plot(x_line, y_line, color="red", linewidth=2, label="Regression line")

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction using Linear Regression")
plt.legend()
plt.tight_layout()
plt.savefig("marks_prediction_plot.png", dpi=150)
print("\nPlot saved as marks_prediction_plot.png")
plt.show()
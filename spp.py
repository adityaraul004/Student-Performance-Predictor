import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic student data
def generate_student_data(num_students=200):
    data = {
        'student_id': range(1, num_students + 1),
        'attendance_rate': np.random.uniform(0.7, 1.0, num_students),
        'participation_score': np.random.uniform(5, 10, num_students),
        'homework_avg': np.random.uniform(60, 100, num_students),
        'midterm_score': np.random.uniform(50, 100, num_students),
        'project_score': np.random.uniform(70, 100, num_students),
        'hours_studied_weekly': np.random.uniform(1, 15, num_students)
    }
    noise = np.random.normal(0, 5, num_students)
    data['final_exam_score'] = (
        15 * data['attendance_rate'] +
        2 * data['participation_score'] +
        0.3 * data['homework_avg'] +
        0.25 * data['midterm_score'] +
        0.2 * data['project_score'] +
        1 * data['hours_studied_weekly'] +
        noise
    )
    data['final_exam_score'] = np.clip(data['final_exam_score'], 0, 100)
    return pd.DataFrame(data)

# Generate the dataset
df = generate_student_data(200)

# Display basic information
print(df.head())
print(df.describe())
print(df.isnull().sum())

# Visualize correlations
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Visualize relationships with final exam score
features = ['attendance_rate', 'participation_score', 'homework_avg', 
            'midterm_score', 'project_score', 'hours_studied_weekly']

plt.figure(figsize=(15, 10))
for i, feature in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sns.scatterplot(x=feature, y='final_exam_score', data=df)
    plt.title(f'{feature} vs final_exam_score')
plt.tight_layout()
plt.show()

# Prepare data for modeling
X = df.drop(['student_id', 'final_exam_score'], axis=1)
y = df['final_exam_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} Performance:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print("-" * 30)
    return mse, rmse, r2

# Evaluate models
lr_metrics = evaluate_model(y_test, lr_pred, "Linear Regression")
rf_metrics = evaluate_model(y_test, rf_pred, "Random Forest")

# Visualize predictions vs actual
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(y_test, lr_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Linear Regression: Actual vs Predicted')

plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Random Forest: Actual vs Predicted')
plt.tight_layout()
plt.show()

# Feature importance for Random Forest
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Feature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# Prediction function
def predict_student_performance(model, attendance_rate, participation_score, 
                               homework_avg, midterm_score, project_score, 
                               hours_studied_weekly):
    student_data = pd.DataFrame({
        'attendance_rate': [attendance_rate],
        'participation_score': [participation_score],
        'homework_avg': [homework_avg],
        'midterm_score': [midterm_score],
        'project_score': [project_score],
        'hours_studied_weekly': [hours_studied_weekly]
    })
    prediction = model.predict(student_data)[0]
    return prediction

# Example usage 
if __name__ == "_main_":
    df = generate_student_data(200)
    X = df.drop(['student_id', 'final_exam_score'], axis=1)
    y = df['final_exam_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    example_prediction = predict_student_performance(
        model,
        attendance_rate=0.9,
        participation_score=8.5,
        homework_avg=85,
        midterm_score=78,
        project_score=92,
        hours_studied_weekly=10
    )
    print(f"Predicted final exam score: {example_prediction:.2f}")
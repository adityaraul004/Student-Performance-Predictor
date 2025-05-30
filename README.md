# Student-Performance-Predictor

A Machine Learning Project for predicting student final exam scores based on academic and behavioral indicators using Linear Regression and Random Forest models.

## 🎯 Project Overview

This project implements a machine learning-based approach to predict student performance using various academic indicators. The system helps in early identification of students who may need academic assistance by analyzing factors such as attendance, participation, homework scores, and study habits.

## 📊 Dataset

- **Size**: 200 synthetic student records
- **Features**: 
  - Attendance Rate (%)
  - Participation Score
  - Homework Average
  - Midterm Score
  - Project Score
  - Weekly Study Hours
- **Target**: Final Exam Score (0-100)

## 🤖 Machine Learning Models

### Implemented Models:
1. **Linear Regression** - Baseline model for linear relationships
2. **Random Forest Regressor** - Primary model (best performance)
3. **Ridge Regression** - Regularized linear model
4. **Gradient Boosting Regressor** - Sequential ensemble method

### Best Performing Model:
- **Random Forest Regressor**
- **RMSE**: 5.78
- **R² Score**: 0.60

## 🛠️ Technologies Used

- **Python 3.x**
- **Scikit-learn** - Machine learning algorithms
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization
- ** Tkinter** - Gui Interface

## ✅Results
![image](https://github.com/user-attachments/assets/e92a542d-4490-4cd1-ade6-d48a51bbce4a)
![image](https://github.com/user-attachments/assets/3ef26489-0d30-459b-bdf4-8bd19d34cf5b)
![image](https://github.com/user-attachments/assets/680c4313-2d21-41ce-862c-2a004d60ebad)
![image](https://github.com/user-attachments/assets/8c67a1ed-1db5-43ca-a61c-04408dff1620)
![image](https://github.com/user-attachments/assets/ae6c70ff-0675-488d-b2fa-115992034668)
![image](https://github.com/user-attachments/assets/6cee25a0-340e-437a-a4aa-45435039116b)
![image](https://github.com/user-attachments/assets/c11e8cb2-2083-4e44-b297-4b3f84a6846e)


## 📁 Project Structure

```
student-performance-predictor/
spp.py - model evaluation 
spp_with_gui - real time predictor gui interface 

## 📈 Model Performance

| Model             |  RMSE |   R² Score | MAE  |
|-------------------|-------|----------|-------|
| Linear Regression | 6.42  |   0.54   | 4.87  |
| **Random Forest** | 5.78  |   0.60   | 4.23  |
| Ridge Regression  | 6.38  |   0.55   | 4.91  |
| Gradient Boosting | 6.01  |   0.58   | 4.45  |

## 🔍 Feature Importance

Top predictive factors (Random Forest):
1. **Attendance Rate** (28.5%)
2. **Midterm Score** (24.2%)
3. **Homework Average** (19.8%)
4. **Study Hours** (15.3%)
5. **Project Score** (8.7%)
6. **Participation Score** (3.5%)

## 📊 Data Preprocessing

- **Missing Value Handling**: Median/Mean imputation
- **Feature Standardization**: StandardScaler for uniform scaling
- **Correlation Analysis**: Multicollinearity detection
- **Train-Test Split**: 80-20 ratio with fixed random state

## 🎯 Key Features

- **Real-time Predictions**: Input student data and get instant score predictions
- **Feature Importance Analysis**: Understand which factors most impact performance
- **Model Comparison**: Multiple algorithms evaluated for best performance
- **Web Interface**: User-friendly Flask application for educators
- **Visualization**: Comprehensive plots for data analysis and model interpretation

## 📋 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
flask>=2.0.0
```

## 🔮 Future Enhancements

- [ ] Integration with real student data
- [ ] Time-series analysis for academic progression
- [ ] Deep learning models (Neural Networks)
- [ ] Mobile application development
- [ ] Advanced feature engineering
- [ ] Automated model retraining pipeline

## 👥 Contributor

- **Aditya Narayan Raul** (1CR22CS009)

## 📄 License

This project is part of academic coursework for the Bachelor of Engineering program at Visvesvaraya Technological University.

---

**Note**: This project uses synthetic data for demonstration purposes. For production use with real student data, ensure compliance with educational data privacy regulations and institutional policies.

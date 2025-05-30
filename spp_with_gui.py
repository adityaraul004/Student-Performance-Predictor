import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import messagebox

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic student data with ULTRA-REALISTIC scoring
def generate_student_data(num_students=500):
    data = {
        'student_id': range(1, num_students + 1),
        'attendance_rate': np.random.uniform(0.0, 1.0, num_students),  # Include 0% attendance
        'participation_score': np.random.uniform(0, 10, num_students),
        'homework_avg': np.random.uniform(0, 100, num_students),  # Include 0% homework
        'midterm_score': np.random.uniform(0, 100, num_students),  # Include 0% midterm
        'project_score': np.random.uniform(0, 100, num_students),  # Include 0% project
        'hours_studied_weekly': np.random.uniform(0, 20, num_students)
    }
    
    # ULTRA-REALISTIC: No base score, purely performance-based
    final_scores = []
    
    for i in range(num_students):
        # NO base score - students must earn every point
        
        # Weighted contributions (must add up to 100% max)
        attendance_weight = 0.20  # 20% of final score
        participation_weight = 0.10  # 10% of final score  
        homework_weight = 0.25  # 25% of final score
        midterm_weight = 0.20  # 20% of final score
        project_weight = 0.15  # 15% of final score
        study_weight = 0.10  # 10% of final score
        
        # Calculate each contribution as percentage of 100
        attendance_contrib = data['attendance_rate'][i] * (attendance_weight * 100)
        participation_contrib = (data['participation_score'][i] / 10) * (participation_weight * 100)
        homework_contrib = (data['homework_avg'][i] / 100) * (homework_weight * 100)
        midterm_contrib = (data['midterm_score'][i] / 100) * (midterm_weight * 100)
        project_contrib = (data['project_score'][i] / 100) * (project_weight * 100)
        study_contrib = min(data['hours_studied_weekly'][i] / 20, 1) * (study_weight * 100)
        
        # Add very small random noise
        noise = np.random.normal(0, 1.5)
        
        # Calculate final score - pure performance based
        final_score = (attendance_contrib + participation_contrib + homework_contrib + 
                      midterm_contrib + project_contrib + study_contrib + noise)
        
        # Ensure realistic bounds (0-100)
        final_score = max(0, min(100, final_score))
        final_scores.append(final_score)
    
    data['final_exam_score'] = final_scores
    
    # Add some guaranteed low performers to train the model properly
    num_low_performers = num_students // 10  # 10% guaranteed low performers
    for i in range(num_low_performers):
        idx = np.random.randint(0, num_students)
        # Create genuinely poor students
        data['attendance_rate'][idx] = np.random.uniform(0.0, 0.3)
        data['participation_score'][idx] = np.random.uniform(0, 2)
        data['homework_avg'][idx] = np.random.uniform(0, 30)
        data['midterm_score'][idx] = np.random.uniform(0, 25)
        data['project_score'][idx] = np.random.uniform(0, 35)
        data['hours_studied_weekly'][idx] = np.random.uniform(0, 2)
        
        # Recalculate their final score
        final_score = (
            data['attendance_rate'][idx] * 20 +
            (data['participation_score'][idx] / 10) * 10 +
            (data['homework_avg'][idx] / 100) * 25 +
            (data['midterm_score'][idx] / 100) * 20 +
            (data['project_score'][idx] / 100) * 15 +
            min(data['hours_studied_weekly'][idx] / 20, 1) * 10 +
            np.random.normal(0, 1)
        )
        data['final_exam_score'][idx] = max(0, min(100, final_score))
    
    return pd.DataFrame(data)

# Prediction function
def predict_student_performance(model, attendance_rate, participation_score, 
                               homework_avg, midterm_score, project_score, 
                               hours_studied_weekly):
    """Predict student performance using the trained model"""
    student_data = pd.DataFrame({
        'attendance_rate': [attendance_rate],
        'participation_score': [participation_score],
        'homework_avg': [homework_avg],
        'midterm_score': [midterm_score],
        'project_score': [project_score],
        'hours_studied_weekly': [hours_studied_weekly]
    })
    prediction = model.predict(student_data)[0]
    return max(0, min(100, prediction))  # Ensure prediction is within 0-100 range

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

# Enhanced GUI function
def launch_gui(model):
    def predict():
        try:
            # Get inputs from entry fields
            attendance = float(entry_attendance.get())
            participation = float(entry_participation.get())
            homework = float(entry_homework.get())
            midterm = float(entry_midterm.get())
            project = float(entry_project.get())
            hours = float(entry_hours.get())
            
            # Validate input ranges
            if not (0.0 <= attendance <= 1.0):
                messagebox.showerror("Input Error", "Attendance Rate should be between 0.0 and 1.0")
                return
            if not (0 <= participation <= 10):
                messagebox.showerror("Input Error", "Participation Score should be between 0 and 10")
                return
            if not (0 <= homework <= 100):
                messagebox.showerror("Input Error", "Homework Average should be between 0 and 100")
                return
            if not (0 <= midterm <= 100):
                messagebox.showerror("Input Error", "Midterm Score should be between 0 and 100")
                return
            if not (0 <= project <= 100):
                messagebox.showerror("Input Error", "Project Score should be between 0 and 100")
                return
            if hours < 0:
                messagebox.showerror("Input Error", "Hours Studied Weekly should be positive")
                return
            
            # Make prediction
            prediction = predict_student_performance(
                model,
                attendance, participation, homework,
                midterm, project, hours
            )
            
            # Determine performance category
            if prediction >= 90:
                category = "Excellent (A)"
                color = "🟢"
            elif prediction >= 80:
                category = "Very Good (B+)"
                color = "🔵"
            elif prediction >= 70:
                category = "Good (B)"
                color = "🟡"
            elif prediction >= 60:
                category = "Average (C)"
                color = "🟠"
            elif prediction >= 50:
                category = "Below Average (D)"
                color = "🔴"
            else:
                category = "Poor (F)"
                color = "⚫"
            
            # Show result with more detailed information
            result_msg = f"""{color} Predicted Final Exam Score: {prediction:.1f}%
Performance Category: {category}

Input Summary:
• Attendance Rate: {attendance:.1%}
• Participation Score: {participation}/10
• Homework Average: {homework}%
• Midterm Score: {midterm}%
• Project Score: {project}%
• Hours Studied Weekly: {hours}

💡 Tips for Improvement:
{get_improvement_tips(attendance, participation, homework, midterm, project, hours)}"""
            
            messagebox.showinfo("Prediction Result", result_msg)
            
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def get_improvement_tips(attendance, participation, homework, midterm, project, hours):
        tips = []
        if attendance < 0.8:
            tips.append("• Improve attendance (target: >80%)")
        if participation < 7:
            tips.append("• Increase class participation")
        if homework < 70:
            tips.append("• Focus on homework completion")
        if hours < 5:
            tips.append("• Increase study time per week")
        if midterm < 70:
            tips.append("• Midterm score is low, focus on your weak areas")
            
        return "\n".join(tips) if tips else "• Keep up the good work!"
    
    # Create GUI window
    root = tk.Tk()
    root.title("Student Performance Predictor")
    root.geometry("600x500")
    root.resizable(False, False)
    
    # Configure colors
    bg_color = "#f0f0f0"
    root.configure(bg=bg_color)
    
    # Add instructions
    instruction_label = tk.Label(root, text="🎓 Student Performance Predictor", 
                                font=("Arial", 16, "bold"), bg=bg_color, fg="#2E7D32")
    instruction_label.grid(row=0, column=0, columnspan=3, pady=15)
    
    sub_instruction = tk.Label(root, text="Enter student data to predict final exam score:", 
                              font=("Arial", 11), bg=bg_color, fg="#424242")
    sub_instruction.grid(row=1, column=0, columnspan=3, pady=(0, 20))
    
    # Labels and entry boxes with helpful hints
    labels_and_hints = [
        ("Attendance Rate", "(0.0 - 1.0, e.g., 0.9 for 90%)"),
        ("Participation Score", "(0 - 10, classroom participation)"), 
        ("Homework Average", "(0 - 100, percentage)"),
        ("Midterm Score", "(0 - 100, percentage)"), 
        ("Project Score", "(0 - 100, percentage)"), 
        ("Hours Studied Weekly", "(hours per week)")
    ]
    
    entries = []
    for i, (label_text, hint) in enumerate(labels_and_hints):
        row_num = i + 2
        
        # Main label
        label = tk.Label(root, text=label_text, font=("Arial", 10, "bold"), 
                        bg=bg_color, anchor="w", width=18, fg="#1976D2")
        label.grid(row=row_num, column=0, padx=(20, 5), pady=(8, 2), sticky="w")
        
        # Hint label
        hint_label = tk.Label(root, text=hint, font=("Arial", 8), 
                             fg="gray", bg=bg_color, anchor="w")
        hint_label.grid(row=row_num, column=1, padx=(5, 5), pady=(8, 2), sticky="w")
        
        # Entry
        entry = tk.Entry(root, width=15, font=("Arial", 10), relief="solid", bd=1)
        entry.grid(row=row_num, column=2, padx=(5, 20), pady=(8, 2), sticky="w")
        entries.append(entry)
    
    # Assign entries to variables for clarity
    entry_attendance, entry_participation, entry_homework, entry_midterm, entry_project, entry_hours = entries
    
    # Add some default values for testing
    entry_attendance.insert(0, "0.85")
    entry_participation.insert(0, "7.5")
    entry_homework.insert(0, "75")
    entry_midterm.insert(0, "70")
    entry_project.insert(0, "80")
    entry_hours.insert(0, "8")
    
    # Button frame
    button_frame = tk.Frame(root, bg=bg_color)
    button_frame.grid(row=12, column=0, columnspan=3, pady=25)
    
    # Predict button
    predict_button = tk.Button(button_frame, text="🎯 Predict Score", command=predict, 
                              bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
                              width=15, height=2, relief="raised", bd=2)
    predict_button.pack(side=tk.LEFT, padx=5)
    
    # Clear button
    def clear_fields():
        for entry in entries:
            entry.delete(0, tk.END)
    
    clear_button = tk.Button(button_frame, text="🗑️ Clear", command=clear_fields, 
                            bg="#ff9800", fg="white", font=("Arial", 10, "bold"),
                            width=12, height=2, relief="raised", bd=2)
    clear_button.pack(side=tk.LEFT, padx=5)
    
    # Sample data button with different scenarios
    def load_sample_data():
        samples = [
            (0.95, 9.0, 92, 88, 95, 12, "High Performer"),
            (0.75, 6.5, 75, 70, 80, 6, "Average Performer"),
            (0.85, 7.5, 82, 75, 85, 8, "Good Performer"),
            (0.50, 3.0, 45, 40, 55, 2, "Struggling Student"),
            (0.30, 2.0, 30, 25, 40, 1, "At-Risk Student"),
            (0.0, 0.0, 0, 0, 0, 0, "Worst Case Scenario")  # Added worst case
        ]
        import random
        sample = random.choice(samples)
        
        entry_attendance.delete(0, tk.END)
        entry_attendance.insert(0, str(sample[0]))
        entry_participation.delete(0, tk.END)
        entry_participation.insert(0, str(sample[1]))
        entry_homework.delete(0, tk.END)
        entry_homework.insert(0, str(sample[2]))
        entry_midterm.delete(0, tk.END)
        entry_midterm.insert(0, str(sample[3]))
        entry_project.delete(0, tk.END)
        entry_project.insert(0, str(sample[4]))
        entry_hours.delete(0, tk.END)
        entry_hours.insert(0, str(sample[5]))
        
        messagebox.showinfo("Sample Loaded", f"Loaded: {sample[6]}")
    
    sample_button = tk.Button(button_frame, text="📝 Sample", command=load_sample_data, 
                             bg="#2196F3", fg="white", font=("Arial", 10, "bold"),
                             width=12, height=2, relief="raised", bd=2)
    sample_button.pack(side=tk.LEFT, padx=5)
    
    # Close button
    def close_app():
        root.quit()
        root.destroy()
    
    close_button = tk.Button(button_frame, text="❌ Close", command=close_app, 
                            bg="#f44336", fg="white", font=("Arial", 10, "bold"),
                            width=12, height=2, relief="raised", bd=2)
    close_button.pack(side=tk.LEFT, padx=5)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", close_app)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("GUI closed by user")
        root.destroy()
    except Exception as e:
        print(f"GUI error: {e}")
        root.destroy()

# Main execution
def main():
    print("Generating realistic student data...")
    # Generate the dataset
    df = generate_student_data(500)
    
    # Display basic information
    print("Dataset Overview:")
    print(df.head())
    print("\nDataset Statistics:")
    print(df.describe())
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Show score distribution
    print(f"\nFinal Exam Score Distribution:")
    print(f"Min: {df['final_exam_score'].min():.1f}")
    print(f"Max: {df['final_exam_score'].max():.1f}")
    print(f"Mean: {df['final_exam_score'].mean():.1f}")
    print(f"Median: {df['final_exam_score'].median():.1f}")
    
    # Prepare data for modeling
    X = df.drop(['student_id', 'final_exam_score'], axis=1)
    y = df['final_exam_score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining models...")
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Evaluate models
    print("\nModel Evaluation:")
    lr_metrics = evaluate_model(y_test, lr_pred, "Linear Regression")
    rf_metrics = evaluate_model(y_test, rf_pred, "Random Forest")
    
    # Feature importance for Random Forest
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Feature Importance (Random Forest):")
    print(feature_importance)
    
    # Test with extreme low values
    print("\nTesting with extreme low values:")
    low_prediction = predict_student_performance(
        rf_model,
        attendance_rate=0.0,
        participation_score=0,
        homework_avg=0,
        midterm_score=0,
        project_score=0,
        hours_studied_weekly=0
    )
    print(f"Extreme low values prediction: {low_prediction:.2f}")
    
    # Test with high values
    print("\nTesting with high values:")
    high_prediction = predict_student_performance(
        rf_model,
        attendance_rate=1.0,
        participation_score=10,
        homework_avg=100,
        midterm_score=100,
        project_score=100,
        hours_studied_weekly=20
    )
    print(f"High values prediction: {high_prediction:.2f}")
    
    # Verify model sanity - if low prediction is still too high, retrain with more extreme data
    if low_prediction > 15:
        print(f"\n⚠️  WARNING: Low prediction ({low_prediction:.1f}) is still too high!")
        print("Adding more extreme low-performance training data...")
        
        # Create additional extreme cases
        extreme_low_data = []
        for _ in range(50):  # Add 50 extreme low cases
            extreme_case = {
                'attendance_rate': np.random.uniform(0.0, 0.1),
                'participation_score': np.random.uniform(0, 1),
                'homework_avg': np.random.uniform(0, 10),
                'midterm_score': np.random.uniform(0, 10),
                'project_score': np.random.uniform(0, 15),
                'hours_studied_weekly': np.random.uniform(0, 1),
                'final_exam_score': np.random.uniform(0, 8)  # Very low scores
            }
            extreme_low_data.append(extreme_case)
        
        extreme_df = pd.DataFrame(extreme_low_data)
        
        # Combine with original data
        combined_df = pd.concat([df, extreme_df], ignore_index=True)
        
        # Retrain model
        X_combined = combined_df.drop(['final_exam_score'], axis=1)
        if 'student_id' in X_combined.columns:
            X_combined = X_combined.drop(['student_id'], axis=1)
        y_combined = combined_df['final_exam_score']
        
        print("Retraining Random Forest with extreme cases...")
        rf_model.fit(X_combined, y_combined)
        
        # Test again
        low_prediction_new = predict_student_performance(
            rf_model,
            attendance_rate=0.0,
            participation_score=0,
            homework_avg=0,
            midterm_score=0,
            project_score=0,
            hours_studied_weekly=0
        )
        print(f"New extreme low values prediction: {low_prediction_new:.2f}")
    
    print(f"\n✅ Model validation complete. Score range: {low_prediction:.1f} - {high_prediction:.1f}")
    
    # Choose the better model
    best_model = rf_model if rf_metrics[2] > lr_metrics[2] else lr_model
    model_name = "Random Forest" if rf_metrics[2] > lr_metrics[2] else "Linear Regression"
    print(f"\nUsing {model_name} model for GUI (R² = {max(rf_metrics[2], lr_metrics[2]):.3f})")
    
    # Launch GUI
    print("\nLaunching GUI...")
    try:
        launch_gui(best_model)
    except Exception as e:
        print(f"Error launching GUI: {e}")

if __name__ == "__main__":
    main()
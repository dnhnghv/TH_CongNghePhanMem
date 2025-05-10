import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# --- Bước 1: Đọc dữ liệu và xử lý NaN ---
df = pd.read_csv('D:\HT\KhoaHocDuLieu\BaiTap\thuchanh6_5\thuchanh6_5\Data_Number_7.csv', sep=';')
+++print("Số giá trị NaN trong mỗi cột:\n", df.isna().sum())

# Điền giá trị trung bình cho các cột số bị thiếu
df.fillna(df.mean(numeric_only=True), inplace=True)

# --- Bước 2: Tạo chỉ số nguy cơ biến chứng ---
df['bmi_std'] = (df['bmi'] - df['bmi'].mean()) / df['bmi'].std()
df['glucose_std'] = (df['blood_glucose'] - df['blood_glucose'].mean()) / df['blood_glucose'].std()
df['hosp_std'] = (df['hospitalizations'] - df['hospitalizations'].mean()) / df['hospitalizations'].std()

# Trọng số: BMI (0.4), đường huyết (0.4), số lần nhập viện (0.2)
df['complication_risk'] = (
    0.4 * df['bmi_std'] +
    0.4 * df['glucose_std'] +
    0.2 * df['hosp_std']
)

# --- Bước 3: Kiểm định chi-squared giữa biến chứng và nhóm tuổi ---
df['age_group'] = pd.cut(df['age'], bins=[0, 40, 60, 120], labels=['<40', '40-60', '>60'])
contingency_table = pd.crosstab(df['age_group'], df['complication'])
chi2, p, _, _ = chi2_contingency(contingency_table)
print(f"\nKiểm định chi-squared: chi2 = {chi2:.2f}, p-value = {p:.4f}")

# --- Bước 4: Mô phỏng xu hướng đường huyết ---
np.random.seed(42)
df['glucose_hist_1'] = df['blood_glucose'] * np.random.uniform(0.9, 1.1, len(df))
df['glucose_hist_2'] = df['blood_glucose'] * np.random.uniform(0.9, 1.1, len(df))

df['glucose_trend'] = np.where(
    (df['glucose_hist_1'] < df['glucose_hist_2']) & (df['glucose_hist_2'] < df['blood_glucose']),
    'Increasing',
    np.where(
        (df['glucose_hist_1'] > df['glucose_hist_2']) & (df['glucose_hist_2'] > df['blood_glucose']),
        'Decreasing',
        'Stable'
    )
)

# --- Bước 5: Xác định mức độ nghiêm trọng ---
df['severity_level'] = np.where(
    (df['blood_glucose'] > 180) & (df['hospitalizations'] >= 2), 'High',
    np.where((df['blood_glucose'] > 140) | (df['hospitalizations'] >= 1), 'Medium', 'Low')
)

# --- Bước 6: Chuẩn bị dữ liệu huấn luyện ---
features = ['age', 'bmi', 'blood_glucose', 'hospitalizations', 'complication_risk']
X = df[features]
y = df['complication']

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xử lý NaN (nếu còn sót lại sau chia tập)
train_data = pd.concat([X_train, y_train], axis=1).dropna()
X_train = train_data[features]
y_train = train_data['complication']

# --- Bước 7: SMOTE - xử lý mất cân bằng ---
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# --- Bước 8: Logistic Regression ---
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_smote, y_train_smote)
y_pred_log = log_reg.predict(X_test)

print("\n=== Hiệu suất Logistic Regression ===")
print(classification_report(y_test, y_pred_log))
print("Ma trận nhầm lẫn:\n", confusion_matrix(y_test, y_pred_log))

# --- Bước 9: Random Forest + Grid Search ---
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_smote, y_train_smote)

best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)

print("\n=== Hiệu suất Random Forest ===")
print("Thông số tốt nhất:", grid_search.best_params_)
print(classification_report(y_test, y_pred_rf))
print("Ma trận nhầm lẫn:\n", confusion_matrix(y_test, y_pred_rf))

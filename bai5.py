import pandas as pd
import numpy as np
from scipy.stats import f_oneway
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Tải dữ liệu
df = pd.read_csv('D:\HT\KhoaHocDuLieu\BaiTap\thuchanh6_5\thuchanh6_5\Data_Number_5.csv', sep=';')

# Nhiệm vụ 1: Tạo chỉ số hiệu suất học tập tổng hợp
# Công thức: Trung bình có trọng số của điểm số (60%) + số giờ tự học (40%)
df['academic_performance'] = (0.2 * df['math_score'] +
                            0.2 * df['literature_score'] +
                            0.2 * df['science_score'] +
                            0.4 * df['study_hours'] * 10)

# Nhiệm vụ 2: Kiểm định ANOVA để đánh giá tác động của tham gia ngoại khóa
def anova_extracurricular():
    low = df[df['extracurricular'] == 'low']['academic_performance']
    medium = df[df['extracurricular'] == 'medium']['academic_performance']
    high = df[df['extracurricular'] == 'high']['academic_performance']
    f_stat, p_value = f_oneway(low, medium, high)
    print(f"Kết quả ANOVA - Thống kê F: {f_stat:.2f}, p-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Tham gia hoạt động ngoại khóa có ảnh hưởng đáng kể đến hiệu suất học tập.")
    else:
        print("Tham gia hoạt động ngoại khóa không có ảnh hưởng đáng kể đến hiệu suất học tập.")

anova_extracurricular()

# Nhiệm vụ 3: Tạo đặc trưng cân bằng học tập
# Công thức: Độ lệch chuẩn của điểm số các môn (thấp = cân bằng tốt)
df['balanced_learning'] = df[['math_score', 'literature_score', 'science_score']].std(axis=1)
df['balanced_learning'] = pd.cut(df['balanced_learning'],
                               bins=[0, 5, 10, float('inf')],
                               labels=['cao', 'trung bình', 'thấp'])

# Nhiệm vụ 4: Tạo đặc trưng rủi ro học tập
# Công thức: Rủi ro cao nếu số buổi vắng mặt > 5 hoặc số giờ tự học < 5
df['academic_risk'] = ((df['absences'] > 5) | (df['study_hours'] < 5)).map({True: 'cao', False: 'thấp'})

# Nhiệm vụ 5 & 6: Phân loại SVM với điều chỉnh siêu tham số
# Chuẩn bị đặc trưng
X = df[['math_score', 'literature_score', 'science_score', 'study_hours', 'absences']]
y = df['academic_risk'].map({'cao': 1, 'thấp': 0})

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Chuẩn hóa đặc trưng
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM với GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

svm = SVC(random_state=42)
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Kết quả
print("\nTham số tốt nhất:", grid_search.best_params_)
print("\nBáo cáo phân loại:")
y_pred = grid_search.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# Hiển thị phân phối hiệu suất học tập
plt.figure(figsize=(8, 6))
plt.hist(df['academic_performance'], bins=20, color='skyblue', edgecolor='black')
plt.title('Phân phối Chỉ số Hiệu suất Học tập')
plt.xlabel('Điểm Hiệu suất Học tập')
plt.ylabel('Tần suất')
plt.savefig('phan_phoi_hieu_suat_hoc_tap.png')
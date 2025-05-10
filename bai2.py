import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import pearsonr

# 1. Đọc dữ liệu từ file CSV
try:
    df = pd.read_csv('D:\HT\KhoaHocDuLieu\BaiTap\thuchanh6_5\thuchanh6_5\HousingData.csv', sep=';')
    df.rename(columns={'MEDV': 'PRICE'}, inplace=True)  # Đổi tên cột MEDV thành PRICE
except Exception as e:
    print(f"Lỗi khi đọc file: {e}")
    exit()

# 2. Khám phá dữ liệu
print(df.info())
print(df.describe())
sns.pairplot(df[['RM', 'LSTAT', 'PTRATIO', 'TAX', 'PRICE']])
plt.show()

# 3. Kiểm tra mối quan hệ Pearson với biến mục tiêu
correlation = {}
for col in df.drop(columns='PRICE').columns:
    df_clean_col = df[[col, 'PRICE']].dropna()
    if len(df_clean_col) > 1:  # Đảm bảo có đủ dữ liệu để tính tương quan
        corr, _ = pearsonr(df_clean_col[col], df_clean_col['PRICE'])
        correlation[col] = corr
    else:
        correlation[col] = 0
print("Tương quan Pearson với PRICE:", pd.Series(correlation).sort_values(ascending=False))

# 4. Xử lý ngoại lai bằng Isolation Forest
iso_forest = IsolationForest(contamination=0.05, random_state=42)
outliers = iso_forest.fit_predict(df.drop(columns='PRICE').fillna(df.mean()))
df_clean = df[outliers == 1]

# 5. Kiểm tra đa cộng tuyến với VIF
X_vif = df_clean.drop(columns='PRICE').fillna(df_clean.mean())
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print("VIF:\n", vif_data)

# Loại bỏ đặc trưng có VIF > 5
high_vif_features = vif_data[vif_data['VIF'] > 5]['feature'].tolist()
if high_vif_features:
    print(f"Loại bỏ các đặc trưng có VIF > 5: {high_vif_features}")
    df_clean = df_clean.drop(columns=high_vif_features)

# 6. Tạo đặc trưng mới
df_clean['room_per_crime'] = df_clean['RM'] / (df_clean['CRIM'] + 1e-6)  # Tránh chia cho 0
df_clean['high_tax'] = (df_clean['TAX'] > df_clean['TAX'].median()).astype(int)
df_clean['RM_LSTAT'] = df_clean['RM'] * df_clean['LSTAT']

# 7. Tạo đặc trưng phi tuyến
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(df_clean[['RM', 'LSTAT']].fillna(df_clean[['RM', 'LSTAT']].mean()))
df_poly = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(['RM', 'LSTAT']))
df_clean = pd.concat([df_clean.reset_index(drop=True), df_poly], axis=1)

# 8. Chuẩn bị dữ liệu huấn luyện
X = df_clean.drop(columns='PRICE').fillna(df_clean.mean())
y = df_clean['PRICE']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 9. Xây dựng pipeline tiền xử lý
numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ])

# 10. Huấn luyện mô hình
models = [
    ('lr', LinearRegression()),
    ('gbr', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
    ('mlp', MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42))
]
results = {}
for name, model in models:
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results[name] = {
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R²': r2_score(y_test, y_pred),
        'MAPE': mean_absolute_percentage_error(y_test, y_pred) * 100
    }
    print(f"{name} - MSE: {mse:.2f}, RMSE: {np.sqrt(mse):.2f}, R²: {r2_score(y_test, y_pred):.2f}, MAPE: {mean_absolute_percentage_error(y_test, y_pred) * 100:.2f}%")

# 11. Kết hợp mô hình với Stacking
stacking_model = StackingRegressor(estimators=models, final_estimator=LinearRegression())
stacking_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('stacking', stacking_model)])
stacking_pipeline.fit(X_train, y_train)
y_pred_stacking = stacking_pipeline.predict(X_test)
mse_stack = mean_squared_error(y_test, y_pred_stacking)
print(f"Stacking - MSE: {mse_stack:.2f}, RMSE: {np.sqrt(mse_stack):.2f}, R²: {r2_score(y_test, y_pred_stacking):.2f}, MAPE: {mean_absolute_percentage_error(y_test, y_pred_stacking) * 100:.2f}%")

# 12. Vẽ residual plot cho mô hình tốt nhất (giả sử Gradient Boosting)
gbr_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', models[1][1])])
gbr_pipeline.fit(X_train, y_train)
y_pred_gbr = gbr_pipeline.predict(X_test)
residuals = y_test - y_pred_gbr
plt.scatter(y_pred_gbr, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot (Gradient Boosting)')
plt.show()

# 13. Phân tích SHAP cho mô hình Gradient Boosting
explainer = shap.Explainer(gbr_pipeline.named_steps['regressor'], preprocessor.fit_transform(X_train))
shap_values = explainer(preprocessor.transform(X_test))
shap.summary_plot(shap_values, X_test, feature_names=X.columns)
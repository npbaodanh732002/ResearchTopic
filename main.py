import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Đọc dữ liệu từ file CSV
data = pd.read_csv('Malware_dataset.csv')

# Tách các đặc trưng và biến mục tiêu
X = data.drop(columns=['hash', 'classification'])  # Thay 'target_column' bằng tên cột của biến mục tiêu
Y = data['classification']

# Khởi tạo mô hình Logistic Regression
model = LogisticRegression()

# Khởi tạo RFE với mô hình và số lượng đặc trưng mong muốn
rfe = RFE(model, n_features_to_select=10)

# Fit RFE với dữ liệu
rfe.fit(X, Y)

# In ra các đặc trưng được chọn
selected_features = X.columns[rfe.support_]
print("Selected features:", selected_features)

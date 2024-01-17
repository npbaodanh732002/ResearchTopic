from process_malware_data import load_data, preprocess_data, train_evaluate_model_dt, train_evaluate_model_nb, train_evaluate_model_knn, train_evaluate_model_lr, train_evaluate_model_rfc
import matplotlib.pyplot as plt

file_path = 'malware_data.csv'
data = load_data(file_path)
X_train, X_test, y_train, y_test = preprocess_data(data)
print(data.head())

# RandomForest
model_rfc, accuracy_rfc = train_evaluate_model_rfc(X_train, X_test, y_train, y_test)
print(f'Random Forest - Accuracy: {accuracy_rfc * 100}%')

# Decision Tree
model_dt, accuracy_dt = train_evaluate_model_dt(X_train, X_test, y_train, y_test)
print(f'Decision Tree - Accuracy: {accuracy_dt * 100}%')

# Naive Bayes
model_nb, accuracy_nb = train_evaluate_model_nb(X_train, X_test, y_train, y_test)
print(f'Naive Bayes - Accuracy: {accuracy_nb * 100}%')

# K-Nearest Neighbors
model_knn, accuracy_knn = train_evaluate_model_knn(X_train, X_test, y_train, y_test)
print(f'K-Nearest Neighbors - Accuracy: {accuracy_knn * 100}%')

# Logistic Regression
model_lr, accuracy_lr = train_evaluate_model_lr(X_train, X_test, y_train, y_test)
print(f'Logistic Regression - Accuracy: {accuracy_lr * 100}%')

# Trực quan hóa độ chính xác
models = ['Random Forest', 'Decision Tree', 'Naive Bayes', 'K-Nearest Neighbors', 'Logistic Regression']
accuracies = [accuracy_rfc*100, accuracy_dt*100, accuracy_nb*100, accuracy_knn*100, accuracy_lr*100]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['#FFD1DC', '#B0E57C', '#FFA07A', '#B19CD9', '#FFD700'])
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.title('Accuracy of Different Models')
plt.show()

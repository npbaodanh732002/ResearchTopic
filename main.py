import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Read the dataset
data = pd.read_csv('Malware_dataset.csv')

# Data preprocessing
data = data.replace(to_replace=r'[!@#$%^&*()_+{}\[\]:;<>,.?~\\|]', value=np.nan, regex=True)
data = data.fillna(value=np.nan)
data['classification'] = data['classification'].map({'benign': 0, 'malware': 1})
data = data.drop(columns=['hash'])

# Split the data
# Split ratio: 80% training, 20% testing
X_train_80, X_test_80, y_train_80, y_test_80 = train_test_split(data.drop(columns=['classification']),
                                                    data['classification'],
                                                    test_size=0.2,
                                                    random_state=42)

# Split ratio: 75% training, 25% testing
X_train_75, X_test_75, y_train_75, y_test_75 = train_test_split(data.drop(columns=['classification']),
                                                              data['classification'],
                                                              test_size=0.25,
                                                              random_state=42)

# Split ratio: 70% training, 30% testing
X_train_70, X_test_70, y_train_70, y_test_70 = train_test_split(data.drop(columns=['classification']),
                                                              data['classification'],
                                                              test_size=0.3,
                                                              random_state=42)

selected_features = ['usage_counter', 'prio', 'normal_prio', 'policy', 'vm_pgoff', 'task_size', 'cached_hole_size',
                     'hiwater_rss', 'nr_ptes', 'last_interval', 'min_flt', 'lock', 'cgtime', 'signal_nvcsw']

print(data)
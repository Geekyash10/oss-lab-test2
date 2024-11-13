#!/usr/bin/env python
# coding: utf-8

# In[45]:


string_list=["yash","vanshul","tejas"]
def reverse(i):
    return i[::-1]
for i in string_list:
    if(len(i)<=5):
        print(reverse(i))
    else:
        continue
            


# In[61]:


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score,f1_score






can_data = load_breast_cancer()
print(load_breast_cancer())
cancer_df = pd.DataFrame(can_data.data, columns=can_data.feature_names)

print("no of missing value",cancer_df.isnull().sum())

if cancer_df.isnull().sum().any():
    cancer_df.fillna(cancer_df.mean(), inplace=True)
    


plt.figure(figsize=(10, 8))
correlation_matrix = cancer_df.corr()
sns.heatmap(correlation_matrix)
plt.title("Correlation Matrix of Breast Cancer Dataset")
plt.show()

knn= KNeighborsClassifier()


X_train, X_test, y_train, y_test = train_test_split(cancer_df, can_data.target, test_size=0.3, random_state=42)


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
















# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings("ignore")


# In[4]:


proj_data = pd.read_csv("C:/Users/hyara/Documents/python1/data sets/diabetes-data.csv")
proj_data


# In[5]:


proj_data.head()


# In[6]:


proj_data.info()


# In[7]:


proj_data.describe().T


# In[8]:


proj_data_copy = proj_data.copy(deep = True)
proj_data_copy


# In[9]:


proj_data_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = proj_data_copy[['Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI']].replace(0,np.NaN)


# In[10]:


print(proj_data_copy.isnull().sum())


# In[11]:


proj_data_copy['Glucose'].fillna(proj_data_copy['Glucose'].mean(), inplace = True)
proj_data_copy['BloodPressure'].fillna(proj_data_copy['BloodPressure'].mean(), inplace = True)
proj_data_copy['SkinThickness'].fillna(proj_data_copy['SkinThickness'].median(), inplace = True)
proj_data_copy['Insulin'].fillna(proj_data_copy['Insulin'].median(), inplace = True)
proj_data_copy['BMI'].fillna(proj_data_copy['BMI'].median(), inplace = True)


# In[12]:


proj_data.shape


# In[18]:


print(proj_data.Outcome.value_counts())
p=proj_data.Outcome.value_counts().plot(kind="bar")
plt.xlabel('diabities')
plt.ylabel("count")


# In[19]:


from sklearn.preprocessing import StandardScaler
scale_X = StandardScaler()
X = scale_X.fit_transform(proj_data_copy.drop(["Outcome"],axis = 1),)
print(X)
X = pd.DataFrame(X,columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])


# In[20]:


y = proj_data_copy.Outcome
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42,
stratify=y)


# In[21]:


from sklearn.neighbors import KNeighborsClassifier
cla = KNeighborsClassifier(n_neighbors=15)
cla.fit(X_train,y_train)


# In[22]:


y_pred = cla.predict(X_test)
y_pred


# In[23]:


from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_pred)
acc


# In[24]:

import pickle
pickle_out = open("classifier.pkl","wb")
pickle.dump(cla, pickle_out)
pickle_out.close()
                  
                  
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)


# In[25]:


a = cla.predict([[6,0,72,35,0,33.6,0.627,50]])

def dia(a):
    if a == 1:
        print("The Patient Has Diabetes")
    else:
        print("The Patient Dont Has Diabetes")
dia(a)


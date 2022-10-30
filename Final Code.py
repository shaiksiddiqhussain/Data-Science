#!/usr/bin/env python
# coding: utf-8

# In[6]:


#Extraction of Data Values from Google Spreadsheet Responses to Python Environment!
#Importing Libraries
import pandas as pd
import numpy as np
import scipy as sp
#Importing Library utility for Access to Google Spreadsheets
import gspread 
#API Credential Keys for Service Account 
from oauth2client.service_account import ServiceAccountCredentials


# In[16]:


data = pd.read_csv('C:/Users/siddi/Documents/100.csv')


# In[13]:


#Checking 'NA' Values..
data.isna() #False
data.isna().sum() #0


# In[358]:


#Removal of columns
data = data.drop(data.columns[[4,5,6,7,8]],axis=1)
data


# In[359]:


#Checking 'NA' Values..
data.isna() #False
data.isna().sum() #0


# In[360]:


data.shape #(12,229)
data.describe()
data.info()
data


# In[361]:


#Separating 'student names' column to a new variable..
names = data.iloc[:,0]
names


# In[362]:


#Applying 1-hot encoding technique..
data = pd.get_dummies(data.iloc[:,1:],dummy_na= False,drop_first=True)
data


# In[363]:


data = pd.concat([names,data], axis=1)
data


# In[364]:


#Scoring Values..
Score = data.iloc[:,1:].sum(axis=1)
Score


# In[365]:


#Combine all into 1 DF..
data = pd.concat([data,Score], axis=1)
data


# In[366]:


data.columns
data.info
print(data.columns.tolist())


# In[367]:


data.rename(columns={ 0: 'Scoring'},inplace=True)
data


# In[368]:


#Calculating Label 'Y' Column..
Prob_values = data['Scoring'] / 228
Prob_values


# In[369]:


#Combining again to 1 DF..
data = pd.concat([data,Prob_values], axis=1)
data


# In[370]:


#Viewing data
data


# In[371]:


#1.Random Forest Regressor
import pandas as pd
import numpy as np

#RandomForestRegressor algorithm
#Assigning Input Column Features(Data) to 'X' and Output Column(Target) to 'Y'..
X=np.array(data.iloc[:,1:231]) #Input Cols
X 
Y=np.array(data.iloc[:,231]) #Output Cols
Y


# In[372]:


#Random Splitting of data..
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25) #Train data size 75% and Test data size 35%
X_train,X_test,Y_train,Y_test


# In[373]:


#Importing some specific packages..
from sklearn.ensemble import RandomForestRegressor

#Random Forest Regressor function..
rfc = RandomForestRegressor(n_estimators=450,bootstrap=True,n_jobs=-1,random_state=42)
rfc


# In[374]:


#Fitting/Training the model..
rfregtrain = rfc.fit(X_train,Y_train)
rfregtrain


# In[375]:


#Prediction the model on test data..
rfregtest = rfregtrain.predict(X_test)
rfregtest


# In[376]:


from sklearn.metrics import accuracy_score 
from sklearn.metrics import mean_squared_error
#Evaluating some confusion matrix,score on test data..
res1 = mean_squared_error(rfregtest,Y_test)
res1
print("The Metric is",res1) #0.0126469
#Accuracy is 98.735%


# In[377]:


#Prediction the model on train data..
rfregtrain = rfregtrain.predict(X_train)
rfregtrain


# In[378]:


#Evaluating some confusion matrix,score on train data..
res2 = mean_squared_error(rfregtrain,Y_train) 
res2
print("The Metric is again",res2) 
#Accuracy is 99.78643%


# In[379]:


#Grid Search CV Technique..
from sklearn.model_selection import GridSearchCV
rfr_grid = RandomForestRegressor(n_estimators=450,bootstrap=True,n_jobs=-1,random_state=42)
rfr_grid


# In[380]:


param_grid = {"max_features": [3,7,9,4,2,1,5], "min_samples_split": [4,5,6,8]}
param_grid


# In[381]:


gs = GridSearchCV(rfr_grid,param_grid,n_jobs=-1,cv=5,scoring="accuracy")
gs


# In[382]:


gstrain = gs.fit(X_train,Y_train)
gstrain


# In[383]:


gstest = gstrain.predict(X_test)
gstest


# In[384]:


gstrain.best_params_
gstrain.best_estimator_ #RandomForestRegressor


# In[385]:


gstrain.best_params_


# In[386]:


from sklearn.metrics import accuracy_score #Importing on evaluation metrics..
from sklearn.metrics import mean_squared_error
#Evaluating some score on test data..
#Error Score Metric..
res3 = mean_squared_error(gstest,Y_test)
res3
print("The Error Score Metric is",res3) 
#Accuracy is 98.74486%


# In[387]:


#Prediction the model on train data..
gstrain = gstrain.predict(X_train)
gstrain


# In[388]:


#Evaluating some score on train data..
#Error Score Metric..
res4 = mean_squared_error(gstrain,Y_train)
res4
print("The Error Score Metric is again",res4) 
#Accuracy is 99.29803927%


# In[389]:


res5 = 1 - res3


# In[390]:


res5


# In[391]:


res6 = 1 - res4


# In[392]:


res6


# In[393]:


#Implementing AutoML Library to automate and picks up the good algorithm by itself!


# In[394]:


#TPOT - Auto ML Library
#pip install tpot
import tpot
from sklearn.model_selection import StratifiedKFold
from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
import os


# In[395]:


get_ipython().system('pip install sklearn fsspec xgboost')
get_ipython().run_line_magic('pip', 'install -U distributed scikit-learn dask-ml dask-glm')
get_ipython().run_line_magic('pip', 'install "tornado>=5"')
get_ipython().run_line_magic('pip', 'install "dask[complete]"')
get_ipython().system('pip install TPOT')


# In[396]:


X=np.array(data.iloc[:,1:231]) #Input Cols
X 
Y=np.array(data.iloc[:,231]) #Output Cols
Y


# In[397]:


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.20) #Train data size 80% and Test data size 20%
X_train,X_test,Y_train,Y_test


# In[398]:


#TPOT REGRESSOR
reg = TPOTRegressor(verbosity=2, population_size=50, generations=10, random_state=35)
reg


# In[408]:


#Fit the regressor on training data
reg.fit(X_train, Y_train)


# In[410]:


#Print the results on test data
print(reg.score(X_test, Y_test))
#Best model given by TPOTRegressor is 'LassoLarsCV'


# In[411]:


#Save the model in an external file..
reg.export('tpotregrssrout.py')


# In[ ]:


#Not enough budget to run the model on this FLAML Automl technique..


# In[421]:


from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


# In[422]:


train,test = train_test_split(data,test_size = 0.30)
train,test


# In[423]:


train_X = train.iloc[:,1:230]
train_Y = train.iloc[:,230]
test_X = test.iloc[:,1:230]
test_Y = test.iloc[:,230]


# In[424]:


#Implemeting Kernel Function..
#Simple linear function
simple_linear = SVR(kernel = "linear")
simple_linear.fit(train_X,train_Y)
testingX = simple_linear.predict(test_X)
testingX


# In[425]:


import numpy 
#Prediction of single record!
ran_data = ['J',1,0,1,1,0,0,1,0,1,0,1,1,0,
            0,1,1,1,1,0,0,1,1,1,1,1,0,
            1,0,1,0,1,0,1,0,1,1,1,0,1,
            1,0,1,1,1,1,1,0,0,1,0,0,1,
            1,1,0,1,1,1,1,1,0,1,1,1,0,
            1,1,0,0,0,0,1,1,0,1,0,1,1,
            0,1,1,0,1,0,1,1,0,0,1,1,0,
            0,1,0,0,1,1,0,1,1,0,0,1,1,
            0,1,0,0,0,0,1,1,1,1,0,1,0,
            1,0,0,1,0,0,1,1,1,1,0,1,1,
            0,1,0,0,1,1,0,0,1,0,1,0,0,
            1,1,1,1,0,0,1,1,0,1,1,0,0,
            1,1,1,0,0,1,0,1,1,0,1,0,1,
            0,0,0,0,1,0,0,1,0,0,1,0,0,
            0,1,0,1,1,0,0,0,1,1,0,1,1,
            1,0,1,0,0,1,0,0,1,0,0,1,1,
            0,0,1,0,1,0,1,1,0,1,0,1,0,
            0,0,1,1,0,1,0,123]
ran_data
ran_data_arr = np.array(ran_data[1:])
ran_data_num = ran_data_arr.reshape(1,-1)
pred_single_rec = simple_linear.predict(ran_data_num)
pred_single_rec


# In[426]:


#Saving the above model for deployment process
#Serialize the model and load it later!
import sklearn.externals 
import joblib
import pickle
#joblib.dump(simple_linear, "simplinear_reg.pkl")
pickle.dump(simple_linear, open('simplinear_reg.pkl','wb'))
#ml_model = pickle.load(open('simplinear_reg.pkl','rb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[427]:





# In[ ]:





# In[ ]:





# In[ ]:





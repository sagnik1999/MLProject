#!/usr/bin/env python
# coding: utf-8
#Used Jupyter Notebook to write the program and converted .ipynb to .py for sending the file
# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
pd.set_option("display.max_rows",None,"display.max_columns",None)


# In[2]:


a=pd.read_csv("train_u6lujuX_CVtuZ9i.csv") #NULL values found using isnull.sum() for cleaning data
import math
median_LAT=math.floor(a.Loan_Amount_Term.median())
median_LA=math.floor(a.LoanAmount.median())
a.Loan_Amount_Term=a.Loan_Amount_Term.fillna(median_LAT)#median Values for missing term 
a.LoanAmount=a.LoanAmount.fillna(median_LA)#median Values for missing amounts 
a.Gender=a.Gender.fillna(np.random.choice((0,1))) #randomised NaN values
a.Married=a.Married.fillna(np.random.choice((0,1)))
a.Dependents=a.Dependents.fillna(0)#Assumed NaN values to be 0
a.Self_Employed=a.Self_Employed.fillna(0)
a.Credit_History=a.Credit_History.fillna(0)
nullc=a.columns[a.isnull().any()] #Cross checked for any NaN values left or not
a[nullc].isnull().sum()


# In[3]:

a.Gender=a.Gender.replace(('Male','Female'),(0,1))
a.Married=a.Married.replace(('Yes','No'),(1,0))
a.Dependents=a.Dependents.replace(('3+',3))
a.Self_Employed=a.Self_Employed.replace(('Yes','No'),(1,0))
a.Property_Area=a.Property_Area.replace(('Rural','Semiurban','Urban'),(0,1,2))
a.Loan_Status=a.Loan_Status.replace(('Y','N'),(1,0))
a.Education=a.Education.replace(('Graduate','Not Graduate'),(1,0))

# In[6]:Training

reg=linear_model.LinearRegression()
reg.fit(a[['Married','Education','Dependents','Self_Employed']],a.LoanAmount)
#Cleaning of test data
b=pd.read_csv("test_Y3wMUE5_7gLdaTN.csv")
median_LAT2=math.floor(b.Loan_Amount_Term.median())
median_LA2=math.floor(b.LoanAmount.median())
b.Loan_Amount_Term=b.Loan_Amount_Term.fillna(median_LAT2)#median Values for missing term 
b.LoanAmount=b.LoanAmount.fillna(median_LA2)#median Values for missing amounts 
b.Gender=b.Gender.fillna(np.random.choice((0,1))) #randomised NaN values
b.Married=b.Married.fillna(np.random.choice((0,1)))
b.Dependents=b.Dependents.fillna(0)#Assumed NaN values to be 0
b.Self_Employed=b.Self_Employed.fillna(0)
b.Credit_History=b.Credit_History.fillna(0)
b.Gender=b.Gender.replace(('Male','Female'),(0,1))
b.Married=b.Married.replace(('Yes','No'),(1,0))
b.Dependents=b.Dependents.replace(('3+',3))
b.Self_Employed=b.Self_Employed.replace(('Yes','No'),(1,0))
b.Property_Area=b.Property_Area.replace(('Rural','Semiurban','Urban'),(0,1,2))
b.Education=b.Education.replace(('Graduate','Not Graduate'),(1,0))
d=reg.predict(b[['Married','Education','Dependents','Self_Employed']])
#d is the dataframe containg the predicted values
#The project objective mentioned the amount of loan a user can take based on marital status,Education,Dependents and Employement
#We can safely use linear regression for multiple varaibles  

#c




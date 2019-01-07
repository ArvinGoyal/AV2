# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 19:22:00 2018

@author: arvgoyal
"""

import os
import pandas as pd
import numpy as np


os.chdir ("D:\Arvin\ArtInt\input files\AV\\02_loan")    ##dohble slsash is needed to except zero in directory name
loan = pd.read_csv ("Loan_Train.csv", header = 0, index_col = 'Loan_ID')

loan.shape
loan.describe()
loan.info()
loan.head(10)

pd.set_option('display.max_columns',12)   
         


####################
#Encoding
####################

loan.info()
loan.Self_Employed.unique()
loan['Education'].unique()

from collections import Counter
Counter(loan.Dependents).most_common(4)


def encod1(x):
    cleanup_nums = {"Gender": {"Female":0, "Male":1},
                    "Married": {"No":0, "Yes":1},
                    "Dependents": {"0":0, "1":1, "2":2, "3+":3},
                    "Education":     {"Not Graduate": 0, "Graduate": 1},
                    "Self_Employed": {"No":0, "Yes":1},                    
                    "Property_Area": {"Rural": 0, "Semiurban": 1, "Urban": 2}
         }
    x.replace(cleanup_nums, inplace=True)
    return x

loan = encod1 (loan)

loan.Loan_Status =  loan.Loan_Status.astype('category')
#################
#EDA
#################
# First we will see if we do have any corelation in varable which can be used for imputaion
loan.corr()

import pylab as pl
pl.matshow (loan.corr())
pl.scatter(loan['Gender'], loan['Credit_History'])
pl.scatter(loan['LoanAmount'], loan['Credit_History'])
import matplotlib.pyplot as plt
plt.boxplot(loan['LoanAmount'])


loan.groupby (['InvoiceNo', 'Description'])['Quantity'].sum()

##   on Anaconda Prompt pip install fancyimpute
from fancyimpute import IterativeImputer    
X_filled_ii = IterativeImputer().fit_transform(X_incomplete)





####################
## Imputation
####################

# Gender-  By Mode
# Married- By Mode
# Dependent - by KNN on Married, Education, Property_Area
# Self_Employed - by KNN on Gender, Married, Education, Property_Area, Dpendents
# Loan Amount- by linear on Married, Education, Property_Area, Self_employed
# Loan_Amt_term - by by linear on Married, Education, Property_Area, Self_employed, Loan Amount
# Cradit_History - By random Forest Gender, Married, Dependednts Education, Self_Employed.....all


loan['Gender'].mode().iloc[0]
loan['Married'].mode().iloc[0]

# ln.info()  give the overall picture of all varables
# ln.Loan_ID.isnull().sum() will give for individual varaible

def imput_mode (x):
    x['Gender'].fillna(value = 1.0, inplace = True)
    x['Married'].fillna(value = 1.0, inplace = True)
    return x
loan = imput_mode (loan)



from sklearn.neighbors import KNeighborsClassifier  
# dependednt imputation
ln = loan[['Married','Education','Property_Area','Dependents']].copy()
ln = ln.dropna()
X = ln.iloc[:,:3]
y = ln.iloc[:,3]
dep_class = KNeighborsClassifier(n_neighbors=5, algorithm='auto') 
dep_class.fit(X,y) 


ln = loan.iloc[:,[1,2,3,10,4]].copy()
ln = ln.dropna()
X = ln.iloc[:,:4]
y = ln.iloc[:,4]
emp_class = KNeighborsClassifier(n_neighbors=5, algorithm='auto') 
emp_class.fit(X,y)


ln = loan[['Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','Property_Area','LoanAmount']].copy()
ln = ln.dropna()
X = ln.iloc[:,:7]
y = ln.iloc[:,7]
from sklearn.linear_model import LinearRegression
ln_amt  = LinearRegression();
ln_amt.fit(X,y)

ln = loan[['Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Property_Area', 'Loan_Amount_Term']].copy()
ln = ln.dropna()
X = ln.iloc[:,:8]
y = ln.iloc[:,8]
from sklearn.linear_model import LinearRegression
ln_trm  = LinearRegression();
ln_trm.fit(X,y)


ln = loan[['Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount', 'Loan_Amount_Term','Property_Area', 'Credit_History']].copy()
ln = ln.dropna()
X = ln.iloc[:,:9]
y = ln.iloc[:,9]
from sklearn.ensemble import RandomForestClassifier
cr_hst = RandomForestClassifier(n_jobs=2, random_state=0)
cr_hst.fit(X, y)
list (zip(X, cr_hst.feature_importances_))


def imput_dep (x):
    x["Dependents"] = np.where(np.isnan(x["Dependents"]), dep_class.predict(x[['Married','Education','Property_Area']]), x['Dependents'])
    return x

def imput_emp (x):
    x["Self_Employed"] = np.where(np.isnan(x["Self_Employed"]), emp_class.predict(x[['Married','Dependents','Education','Property_Area']]), x['Self_Employed'])
    return x

def imput_ln_amt (x):
    x['LoanAmount'] = np.where(np.isnan(x["LoanAmount"]), ln_amt.predict(x[['Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','Property_Area']]), x['LoanAmount'])
    return x

def imput_ln_trm (x):
    x['Loan_Amount_Term'] = np.where(np.isnan(x["Loan_Amount_Term"]), ln_trm.predict(x[['Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount', 'Property_Area']]), x['Loan_Amount_Term'])
    return x    


def imput_cr_hst(x):
    x['Credit_History'] = np.where(np.isnan(x["Credit_History"]), cr_hst.predict(x[['Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount', 'Loan_Amount_Term', 'Property_Area']]), x['Credit_History'])
    return x


def imput_all (x):
    x = imput_mode (x)
    x = imput_dep (x)
    x = imput_emp (x)
    x = imput_ln_amt (x)
    x = imput_ln_trm (x)
    x = imput_cr_hst (x)
    return x

loan = imput_all (loan)

loan.info()

####################
## outleir and scaling is not needed
####################

####################
## test train split and model
####################

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(loan.iloc[:,:11], loan.iloc[:,11], test_size = 0.25, random_state = 0)

# Create a random forest Classifier.
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(X_train, y_train)

list (zip(loan.iloc[:,:12], clf.feature_importances_))

pred_ln = clf.predict(X_test) 
prob = clf.predict_proba(X_test)

pd.crosstab(y_test, pred_ln, rownames=['Actual laon status'], colnames=['Predicted laon status'])


# accuracy around .8
# Precision= .83
#Recall = .89

##  working on actual test data
loan_t = pd.read_csv ("Loan_Test.csv", header = 0, index_col = 'Loan_ID')

loan_t = encod1 (loan_t)
loan_t = imput_all (loan_t)
#loan_t_pred = clf.predict(loan_t)
loan_t ['loan_t_pred']= clf.predict(loan_t)

laon_sub['loan_t_pred'] = loan_t ['loan_t_pred']
loan_t.to_csv('out.csv')

## my score for this submission is 77





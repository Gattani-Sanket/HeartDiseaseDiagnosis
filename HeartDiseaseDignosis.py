# Heart database
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sn

# Importing the dataset
dataset = pd.read_csv('F:\heart.csv')


# data preprocessing

#  removing education
dataset=dataset.drop(["education"],axis=1)

#total null values in a dataset
dataset.isnull().sum()

#removing rows having null value
dataset.dropna(axis=0,inplace=True)

dataset["prevalentStroke"].value_counts()

def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
        ax.set_title(feature+" Distribution",color='DarkRed')
        
    fig.tight_layout()  
    plt.show()
draw_histograms(dataset,dataset.columns,6,3)

sn.countplot(x='TenYearCHD',data=dataset)

# determining the characteristics
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,14].values

#backward elimination
import statsmodels.regression.linear_model as sm
X=np.append(arr=np.ones((3751,1)).astype(int),values=X,axis=1)
X_opt=X[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,2,3,4,5,6,7,8,9,10,11,13,14]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,2,4,5,6,7,8,9,10,11,13,14]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,2,4,5,6,7,9,10,11,13,14]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,2,4,5,6,7,9,10,11,14]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,2,4,5,6,7,10,11,14]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,2,4,6,7,10,11,14]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,2,4,6,7,10,14]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt=X[:,[0,1,2,4,6,10,14]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()


#seperating test and train data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_opt,y,test_size=0.2,random_state=0)


#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)

#Applying classifications model and testing the accuracy
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)


#predicting the result for classifiv
y_pred=classifier.predict(x_test)


from sklearn import metrics

cm=metrics.confusion_matrix(y_test,y_pred)
sn.heatmap(cm,annot=True,fmt="d",center=0)

print("Accuracy: ",metrics.accuracy_score(y_test, y_pred))
print("Logistic Regression")


#using K-NN algorithm
from sklearn.neighbors import KNeighborsClassifier
classifierKNN=KNeighborsClassifier(n_neighbors=50)
classifierKNN.fit(x_train,y_train)

y_predKNN=classifierKNN.predict(x_test)



cmKNN=metrics.confusion_matrix(y_test,y_predKNN)
sn.heatmap(cmKNN,annot=True,fmt="d",center=0)
print("Accuracy: ",metrics.accuracy_score(y_test, y_predKNN))
print("K-NN")

#using SVM algorithm
from sklearn.svm import SVC
classifierSVC=SVC(random_state=0)
classifierSVC.fit(x_train,y_train)

y_predSVC=classifierSVC.predict(x_test)



cmsvm=metrics.confusion_matrix(y_test,y_predSVC)
sn.heatmap(cmsvm,annot=True,fmt="d",center=0)
print("Accuracy: ",metrics.accuracy_score(y_test, y_predSVC))
print("SVM")

#using Navie_Bayes
from sklearn.naive_bayes import GaussianNB
classifierNavie=GaussianNB()
classifierNavie.fit(x_train,y_train)

y_predNavie=classifierNavie.predict(x_test)
cmnavie=metrics.confusion_matrix(y_test,y_predNavie)
sn.heatmap(cmnavie,annot=True,fmt="d",center=0)
print("Accuracy: ",metrics.accuracy_score(y_test, y_predNavie))
print("Navie Bayes")

#using Decision_Tree
from sklearn.tree import DecisionTreeClassifier
classifierDTC=DecisionTreeClassifier(criterion="entropy",random_state=0)
classifierDTC.fit(x_train,y_train)

y_predDTC=classifierDTC.predict(x_test)


cmDTC=metrics.confusion_matrix(y_test, y_predDTC)
sn.heatmap(cmDTC,annot=True,fmt="d",center=0)
print("Accuracy: ",metrics.accuracy_score(y_test, y_predDTC))
print("Decision Tree")

#using random forest
from sklearn.ensemble import RandomForestClassifier
classifierRFC=RandomForestClassifier(n_estimators=50,random_state=0)
classifierRFC.fit(x_train,y_train)

y_predRFC=classifierRFC.predict(x_test)

cmRFC=metrics.confusion_matrix(y_test, y_predRFC)
sn.heatmap(cmRFC,annot=True,fmt="d",center=0)

print("Accuracy",metrics.accuracy_score(y_test, y_predRFC))
print("Random Forest")





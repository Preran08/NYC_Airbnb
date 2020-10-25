# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:58:47 2020

@author: NISARG
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import utils

df = pd.read_csv("/home/preran/Downloads/AB_NYC_2019.csv")
print(df.head())

#------------------------------------------------------------------
#Cleaning of Data

print(df.isnull().sum()) #Fing all the null values in the data

#Since only 16 values are null, we would be using forward and backward filling respectively.
df["name"] = df["name"].fillna(method = "ffill")
df["host_name"] = df["host_name"].fillna(method = "bfill")

#Since, it has decimal values, we would be using mean of the column to fillup the values.
df["reviews_per_month"] = df["reviews_per_month"].fillna(df["reviews_per_month"].mean())

#dropping columns that are not significant or could be unethical to use for our future data exploration and predictions
#We are removing last review becasue of too many unknown variables and it is a data time type, 
#Hence, it would be difficult to predict 10052 values in this column.
df.drop(["name","host_name","id","last_review"], axis = 1, inplace = True)


#Dimensions of data
print('Rows     :',df.shape[0])
print('Columns  :',df.shape[1])
print('\nFeatures :\n     :',df.columns.tolist())


#---------------------------------------------------------------------
#Data Visualisation

#Placements of the Hotels
plt.figure(figsize=(8,8))
df['neighbourhood_group'].value_counts().plot.pie(explode=[0,0.1,0,0,0],autopct='%1.1f%%',shadow=True)
plt.title('Share of Neighborhood')
plt.show()

#Spread of Hotels according to latitude and longitude
#colors = df["neighbourhood_group"].unique().count()
plt.figure(figsize=(15,12))
sns.lmplot(x = "longitude", y = "latitude",hue = "neighbourhood_group", data = df)
#plt.scatter(df.longitude,df.latitude, c = colors)
sns.plt.show()

#Variations of prices of hotels with area
plt.figure(figsize=(10,6))
sns.distplot(df[df.neighbourhood_group=='Manhattan'].price,color='maroon',hist=False,label='Manhattan')
sns.distplot(df[df.neighbourhood_group=='Brooklyn'].price,color='black',hist=False,label='Brooklyn')
sns.distplot(df[df.neighbourhood_group=='Queens'].price,color='green',hist=False,label='Queens')
sns.distplot(df[df.neighbourhood_group=='Staten Island'].price,color='blue',hist=False,label='Staten Island')
plt.title('Borough wise price distribution for price<2000')
plt.xlim(0,2000)
plt.show()

#Boxplot showing price variation
ng = df[df.price <500]
plt.figure(figsize=(10,6))
sns.boxplot(y="price",x ='neighbourhood_group' ,data = ng)
plt.title("neighbourhood_group price distribution < 500")
plt.show()

#Type of Rooms:
f,ax=plt.subplots(1,2,figsize=(18,8))
df['room_type'].value_counts().plot.pie(explode=[0,0.05,0],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Share of Room Type')
ax[0].set_ylabel('Room Type Share')
sns.countplot('room_type',data=df,ax=ax[1],order=df['room_type'].value_counts().index)
ax[1].set_title('Share of Room Type')
plt.show()


#Rooms Vs Neighbourhood_Group
plt.figure(figsize=(10,6))
sns.countplot(x = 'room_type',hue = "neighbourhood_group",data = df)
plt.title("Room types occupied by the neighbourhood_group")
plt.show()

#Distribution of Room_Availability and minimum number of nights:
sns.distplot(df[(df['minimum_nights'] <= 30) & (df['minimum_nights'] > 0)]['minimum_nights'], bins=31)
plt.show()

#---------------------------------------------------------------------
#Feature Selection
corrmatrix = df.corr()
f, ax = plt.subplots(figsize=(10,8))
sns.heatmap(corrmatrix, vmax=0.8, square=True)
sns.set(font_scale=0.8)

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import scale, StandardScaler, RobustScaler

#Handling Categorical Variables
df['neighbourhood_group']= df['neighbourhood_group'].astype("category").cat.codes
df['neighbourhood'] = df['neighbourhood'].astype("category").cat.codes
df['room_type'] = df['room_type'].astype("category").cat.codes
df['price_log'] = np.log(df.price+1)


df.drop(['host_id','price'], axis=1, inplace=True)
    
nyc_model_x, nyc_model_y = df.iloc[:,:-1], df.iloc[:,-1]
scaler = StandardScaler()
nyc_model_x = scaler.fit_transform(nyc_model_x)


X_train, X_test, y_train, y_test = train_test_split(nyc_model_x, nyc_model_y, test_size=0.3, random_state=101)

print(utils.multiclass.type_of_target(X_train))
utils.multiclass.type_of_target(y_train.astype("int"))
print(utils.multiclass.type_of_target(X_test))
utils.multiclass.type_of_target(y_test.astype("int"))

#-------------------------------------------------------------------------------
#PCA and Linear Regression Implmentation
from sklearn.decomposition import PCA

pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_
print(explained_variance)

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

from sklearn.linear_model import LinearRegression

classifier = LinearRegression()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
utils.multiclass.type_of_target(y_pred.astype("int"))

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print("---------------------Linear Regression Algorithm-----------------------------")
print("""
        Mean Squared Error: {}
        R2 Score: {}
        Mean Absolute Error: {}
     """.format(
        np.sqrt(mean_squared_error(y_test, y_pred)),
        r2_score(y_test,y_pred),
        mean_absolute_error(y_test,y_pred)
        ))


#--------------------------------------------------------------------------------
#LASSO
from sklearn.linear_model import Lasso
def lasso_reg(input_x, input_y, cv=5):
    ## Defining parameters
    model_Lasso= Lasso()

    # prepare a range of alpha values to test
    alphas = np.array([1,0.1,0.01,0.001,0.0001,0])
    normalizes= ([True,False])

    ## Building Grid Search algorithm with cross-validation and Mean Squared Error score.

    grid_search_lasso = GridSearchCV(estimator=model_Lasso,  
                         param_grid=(dict(alpha=alphas, normalize= normalizes)),
                         scoring='neg_mean_squared_error',
                         cv=cv,
                         n_jobs=-1)

    ## Lastly, finding the best parameters.

    grid_search_lasso.fit(input_x, input_y)
    best_parameters_lasso = grid_search_lasso.best_params_  
    best_score_lasso = grid_search_lasso.best_score_ 
    print(best_parameters_lasso)
    print(best_score_lasso)

kfold_cv=KFold(n_splits=5, random_state=42, shuffle=False)
for train_index, test_index in kfold_cv.split(nyc_model_x,nyc_model_y):
    X_train, X_test = nyc_model_x[train_index], nyc_model_x[test_index]
    y_train, y_test = nyc_model_y[train_index], nyc_model_y[test_index]

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
Lasso_model = Lasso(alpha = 0.001, normalize =False)
Lasso_model.fit(X_train, y_train)
pred_Lasso = Lasso_model.predict(X_test)
print('------------------Phase-1 LASSO-----------------------------')
print("""
        Mean Squared Error: {}
        R2 Score: {}
        Mean Absolute Error: {}
     """.format(
        np.sqrt(mean_squared_error(y_test, pred_Lasso)),
        r2_score(y_test,pred_Lasso),
        mean_absolute_error(y_test,pred_Lasso)
        ))

'''

#---------------------------------------------------------------------------
#Gradient Boosting Algorithm
from sklearn.ensemble import  GradientBoostingRegressor
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01)
GBoost.fit(X_train,y_train)

predicts2 = GBoost.predict(X_test)

print("---------------------Gradient Boosting---------------------------")
print("""
        Mean Squared Error: {}
        R2 Score: {}
        Mean Absolute Error: {}
     """.format(
        np.sqrt(mean_squared_error(y_test, predicts2)),
        r2_score(y_test,predicts2),
        mean_absolute_error(y_test,predicts2)
        ))

#-----------------------------------------------------------------------------
#RandomForest Algorithm
from sklearn.ensemble import RandomForestRegressor
model_name = "Random Forest Regressor"

model_random = RandomForestRegressor(n_estimators = 100, max_depth=15, random_state = 42)
model_random.fit(X_train, y_train)

y_pred_3 = model_random.predict(X_test)
print("----------------Random Forest Algorithm-----------------------")
print("""
        Mean Squared Error: {}
        R2 Score: {}
        Mean Absolute Error: {}
     """.format(
        np.sqrt(mean_squared_error(y_test, y_pred_3)),
        r2_score(y_test,y_pred_3),
        mean_absolute_error(y_test, y_pred_3)
        ))

#-----------------------------------------------------------------------------
#Neural Netwrok Implementation

from sklearn.neural_network import MLPRegressor

regressor = MLPRegressor(hidden_layer_sizes = (100, 75, 50, 25), activation = 'relu', solver = 'sgd',
                                 learning_rate = 'adaptive', alpha = 0.01, random_state = 101)
regressor.fit(X_train, y_train)
#print(regressor.loss_)

y_pred_4 = regressor.predict(X_test)
print("------------------Neural Network----------------------------")
print("""
        Mean Squared Error: {}
        R2 Score: {}
        Mean Absolute Error: {}
     """.format(
        np.sqrt(mean_squared_error(y_test, y_pred_4)),
        r2_score(y_test,y_pred_4),
        mean_absolute_error(y_test,y_pred_4)
        ))
'''
#-----------------------------------------------------------------------------
#Decision Tree Algorithm
from sklearn.tree import DecisionTreeRegressor

decision_model = DecisionTreeRegressor(random_state = 0, max_depth = 10)
decision_model.fit(X_train, y_train)

y_pred_5 = decision_model.predict(X_test)
print("------------------Decision Trees----------------------------")
print("""
        Mean Squared Error: {}
        R2 Score: {}
        Mean Absolute Error: {}
     """.format(
        np.sqrt(mean_squared_error(y_test, y_pred_5)),
        r2_score(y_test,y_pred_5),
        mean_absolute_error(y_test,y_pred_5)
        ))


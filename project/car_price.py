#Import dependencies
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

#Data collection and processing 
#loading the data from csv file to pandas dataframe
car_dataset = pd.read_csv(r'C:\Arslan\VT\project\car_data.csv')
print(car_dataset)

#checking no of rows and columns 
car_dataset.shape

#getting information about the dataset
car_dataset.info

#checking no of missing values
car_dataset.isnull().sum()

#cheking the distribution of categorical data
print(car_dataset.fuel.value_counts())
print(car_dataset.seller_type.value_counts())
print(car_dataset.transmission.value_counts())

#encoding the categorical data
#fuel type
car_dataset.replace({'fuel':{'Petrol':0,'Diesel':1,'CNG':2,'LPG':3,'Electric':4}},inplace=True)
#seller type
car_dataset.replace({'seller_type':{'Dealer':0,'Individual':1,'Trustmark Dealer':2}},inplace=True)
#Transmission type
car_dataset.replace({'transmission':{'Manual':0,'Automatic':1}},inplace=True)
#owner type
car_dataset.replace({'owner':{'First Owner':0,'Second Owner':1,'Third Owner':2,'Fourth & Above Owner':3,'Test Drive Car':4}},inplace=True)
print(car_dataset)

#splitting data into data and target
X=car_dataset.drop(['name','selling_price'],axis=1)
Y=car_dataset['selling_price']
print(X)
print(Y)

#splitting training and test data
X_train , X_test , Y_train , Y_test =train_test_split(X,Y , test_size=0.1,random_state=2)

#model training
#Linear Regression
lin_reg_model=LinearRegression()
lin_reg_model.fit(X_train,Y_train)

#prediction on training data
training_data_prediction=lin_reg_model.predict(X_train)
#R squard error
error_score=r2_score(Y_train,training_data_prediction)
print("R squared Error :",error_score)
#visualize the actual price and predicted price
plt.scatter(Y_train,training_data_prediction)
plt.xlabel("Actual prices")
plt.ylabel("Predicted price")
plt.title("Actual prices vs Predicted prices")
plt.savefig(r'C:\Arslan\VT\project\linear_regression_train.png')
plt.close()

#prediction on test data
test_data_prediction=lin_reg_model.predict(X_test)
#R squard error
error_score=r2_score(Y_test,test_data_prediction)
print("R squared Error :",error_score)
#visualize the actual price and predicted price
plt.scatter(Y_test,test_data_prediction)
plt.xlabel("Actual prices")
plt.ylabel("Predicted price")
plt.title("Actual prices vs Predicted prices")
plt.savefig(r'C:\Arslan\VT\project\linear_regression_test.png')
plt.close()

#linear regression is for all those data that are directly interconnected
#Lasso Regression
lass_reg_model=Lasso()
lass_reg_model.fit(X_train,Y_train)

#prediction on training data
training_data_prediction=lass_reg_model.predict(X_train)
#R squard error
error_score=r2_score(Y_train,training_data_prediction)
print("R squared Error :",error_score)
#visualize the actual price and predicted price
plt.scatter(Y_train,training_data_prediction)
plt.xlabel("Actual prices")
plt.ylabel("Predicted price")
plt.title("Actual prices vs Predicted prices")
plt.savefig(r'C:\Arslan\VT\project\lasso_regression_train.png')
plt.close()

#prediction on test data
test_data_prediction=lass_reg_model.predict(X_test)
#R squard error
error_score=r2_score(Y_test,test_data_prediction)
print("R squared Error :",error_score)
#visualize the actual price and predicted price
plt.scatter(Y_test,test_data_prediction)
plt.xlabel("Actual prices")
plt.ylabel("Predicted price")
plt.title("Actual prices vs Predicted prices")
plt.savefig(r'C:\Arslan\VT\project\lasso_regression_test.png')
plt.close()

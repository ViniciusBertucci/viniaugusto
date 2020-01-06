#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%% Import csv file
data = pd.read_csv('column_2C_weka.csv')
print(plt.style.available)
plt.style.use('ggplot')
data.head()
#%% ploting scatter_matrix
#green = normal
#red = abnormal
#c = color
#figsize: figure size
#diagonal: histohram of each features
#s: size of maker
#marker: marker type
color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'class'],
                                       c=color_list,
                                       figsize= [15,15],
                                       diagonal='hist',
                                       alpha=0.5,
                                       s = 200,
                                       marker = '*',
                                       edgecolor= "black")
plt.show()
sns.countplot(x='class', data = data)
data.loc[:,'class'].value_counts()
#%%% KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
x,y = data.loc[:,data.columns != 'class'], data.loc[:,'class']
knn.fit(x,y)
prediction = knn.predict(x)
print('Prediction: {}'.format(prediction))
print('With KNN (K=3) and without test and train split, the accuracy is', knn.score(x,y))

#%% Train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 1)
knn = KNeighborsClassifier(n_neighbors = 3)
x,y = data.loc[:, data.columns != 'class'], data.loc[:,'class']
knn.fit(x_train,y_train)
prediction = knn.predict(x_test)
print('Prediction: {}'.format(prediction))
print('With KNN (K=3 accuracy is', knn.score(x_test,y_test))
#%% Model complexity
neig = np.arange(1, 25)
train_accuracy = []
test_accuracy = []
# Loop over different values of k
for i, k in enumerate(neig):
    # k from 1 to 25(exclude)
    knn = KNeighborsClassifier(n_neighbors=k)
    # Fit with knn
    knn.fit(x_train,y_train)
    #train accuracy
    train_accuracy.append(knn.score(x_train, y_train))
    # test accuracy
    test_accuracy.append(knn.score(x_test, y_test))

# Plot
plt.figure(figsize=[13,8])
plt.plot(neig, test_accuracy, label = 'Testing Accuracy')
plt.plot(neig, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.title('-value VS Accuracy')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.xticks(neig)
plt.savefig('graph.png')
plt.show()
print("Best accuracy is {} with K = {}".format(np.max(test_accuracy),1+test_accuracy.index(np.max(test_accuracy))))

#%%REGRESSION
        #sacral_slope x pelvic_incidence
data1 = data[data['class']=='Abnormal']
x = np.array(data1.loc[:,'pelvic_incidence']).reshape(-1,1)
y = np.array(data1.loc[:,'sacral_slope']).reshape(-1,1)
plt.figure(figsize=[10,10])
plt.scatter(x,y)
plt.xlabel('pelvic_incidenc')
plt.ylabel('sacral_slope')
plt.show()

        #lumbar_lordosis_angle x pelvic_incidence
data2 = data[data['class']=='Abnormal']
c = np.array(data2.loc[:,'lumbar_lordosis_angle']).reshape(-1,1)
d = np.array(data2.loc[:,'pelvic_incidence']).reshape(-1,1)
plt.figure(figsize=[10,10])
plt.scatter(c,d,color='blue')      
plt.xlabel('lumbar_lordosis_angle')
plt.ylabel('sacral_slope')
plt.show()
 
        #pelvic_incidence x lumbar_lordosis_angle
data3 = data[data['class']=='Abnormal']
f = np.array(data3.loc[:,'pelvic_incidence']).reshape(-1,1)
g = np.array(data3.loc[:,'lumbar_lordosis_angle']).reshape(-1,1)
plt.figure(figsize=[10,10])
plt.scatter(f,g,color='green')      
plt.xlabel('pelvic_incidence')
plt.ylabel('lumbar_lordosis_angle')
plt.show()        

        #pelvic_incidence x pelvic_radius
data4 = data[data['class']=='Abnormal']
h = np.array(data4.loc[:,'pelvic_incidence']).reshape(-1,1)
i = np.array(data4.loc[:,'pelvic_radius']).reshape(-1,1)
plt.figure(figsize=[10,10])
plt.scatter(h,i,color='black')      
plt.xlabel('pelvic_incidence')
plt.ylabel('pelvic_radius')
plt.show()        

#%% Linear regression
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
#Predict space
predict_space = np.linspace(min(x), max(y)).reshape(-1,1)
#Fit
reg.fit(x,y)
#Predict
predicted = reg.predict(predict_space)
#R^2
plt.plot(predict_space, predicted, color='black', linewidth=3)
plt.scatter(x,y)
plt.xlabel('pelvic_incidence')
plt.ylabel('sacral_slope')
plt.show()
print('R^2 score: ', reg.score(x,y))
#%%Cross Validation
from sklearn.model_selection import cross_val_score
reg = LinearRegression()
k = 5
cv_result = cross_val_score(reg,x,y,cv=k) # uses R^2 as score 
print('CV Scores: ',cv_result)
print('CV scores average: ',np.sum(cv_result)/k)

#Ridge
from sklearn.linear_model import Ridge
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 2, test_size = 0.3)
ridge = Ridge(alpha = 0.1, normalize = True)
ridge.fit(x_train,y_train)
ridge_predict = ridge.predict(x_test)
print('Ridge score: ',ridge.score(x_test,y_test))

#Lasso
from sklearn.linear_model import Lasso
x = np.array(data1.loc[:,['pelvic_incidence','pelvic_tilt numeric','lumbar_lordosis_angle','pelvic_radius']])
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 3, test_size = 0.3)
lasso = Lasso(alpha = 0.1, normalize = True)
lasso.fit(x_train,y_train)
ridge_predict = lasso.predict(x_test)
print('Lasso score: ',lasso.score(x_test,y_test))
print('Lasso coefficients: ',lasso.coef_)

#%% Confusion matrix with random forest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
x,y = data.loc[:,data.columns !='class'], data.loc[:, 'class']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 1)
rf = RandomForestClassifier(random_state = 4)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('Confusion matrix: \n', cm)
print('Classification report: \n', classification_report(y_test, y_pred))

sns.heatmap(cm, annot=True, fmt='d')
plt.show

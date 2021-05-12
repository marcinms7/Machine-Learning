"""
@author: marcinswierczewski
"""
import Regressions
import Clustering
import Bayes_Classifier 
from sklearn import datasets
import numpy as np
import pandas as pd

# Applying Bayes Classifier with my algorithm
iris = datasets.load_iris()
X = iris.data[:, :2]  
y = iris.target
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.4,random_state=42)
Bayes = Bayes_Classifier.Bayes_classifier(xtrain , ytrain)
pi= Bayes.prior()
mi,sigma =  Bayes.class_conditional_density()
predict =  Bayes.plugin_classifier(xtest , mi, sigma, pi,3)

# testing - comparing to model from sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
model = GaussianNB()
model.fit(xtrain,ytrain)
ypredd = model.predict(xtest)

# Clustering
# Applying my clustering algorithm
from sklearn.datasets import load_wine
data = load_wine()
X = data.data
# X = X[:,:5]

Clustering.k_means(X)


# Comparing to sklearn algorithm
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
kmeans.labels_

# Running my clustering algorithm. This may not work sometimes 
# due to randomness - matrix cannot be singular

try:
    output = Clustering.gaussian_mixture_model(X, clusters=5,iterations=15)
except:
    try:
        output =  Clustering.gaussian_mixture_model(X, clusters=5,iterations=15)
    except:
        output =  Clustering.gaussian_mixture_model(X, clusters=5,iterations=15)

print(output)

# Regressions
# Applying my regression algorithms using OLS
# Note that function splits training and test inside, after declaring the dependant and independant variables

df = pd.read_csv('/Users/marcinswierczewski/Downloads/hungary_chickenpox/hungary_chickenpox.csv')
X = df.iloc[:,2:].values
y = df.iloc[:,1].values

Reg = Regressions.Regressions(X,y,0.5)

Reg.OLS()
# # Tests of accuracy 
X_tra = X[261:]
X_tes = X[:261]
y_tra = y[261:]
y_tes = y[:261]
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

regression = LinearRegression()
linear_model = regression.fit(X_tra,y_tra)
print(linear_model.coef_)
y_predd = linear_model.predict(X_tes)
(sum(np.abs(y_tes - y_predd)))/len(y_tes)
mean_absolute_error(y_tes, y_predd)


# Lasso regression - requires some work
# values  = [0.1,0.5,1,2,5]
# for k in values:
#     print(Reg.Lasso(k))



# Logit - requires some work

Logistic = Regressions.Logit()
train = Logistic.train(X_tra,y_tra,100)
Logistic.predict(X_tes,train)
Logistic.classify(X_tes,train)





from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_tra,y_tra)
preds2 = reg.predict(X_tes)






import numpy as np

class Regressions:
    
    def __init__(self,  X,y, split):
        self.split = split
        self.y = y
        self.X = X
        lenn = len(X)
        split = round(len(X) * split)
        self.train_X = X[split:]
        self.test_X = X[:split]
        self.train_y = y[split:]
        self.test_y = y[:split]

    def Lasso (self,lmbda, predict = True, standarize = True):
        X = self.train_X
        y = self.train_y
        
        
        X_test = self.test_X
        y_test = self.test_y
        
        if standarize:
            
            xmean =  np.mean(X, axis=0)
            ymean = np.mean(y, axis=0)
            xsigma = np.std(X, axis=0)
            ysigma = np.std(y, axis=0)
            
            X = (X - xmean)/xsigma
            y = (y - ymean)/ysigma
            X_test = (X_test - xmean)/xsigma
            y_test = (y_test - xmean)/xsigma
            
        
        A = (X.T.dot(X))
        I = np.identity(len(A))
        lmbda = lmbda * I
        A = np.linalg.inv(A + lmbda)
        betas = A.dot(X.T).dot(y)
        print(betas)
        if predict == False:
            return betas
        else:
            ypred = X_test.dot(betas)
            error = (sum(np.abs(y_test - ypred)))/len(y_test)
            return error
            
        
    def OLS (self, predict = True,standarize = False):
        X = self.train_X
        y = self.train_y
        X_test = self.test_X
        y_test = self.test_y
        
        if standarize:
            
            xmean =  np.mean(X, axis=0)
            ymean = np.mean(y, axis=0)
            xsigma = np.std(X, axis=0)
            ysigma = np.std(y, axis=0)
            
            X = (X - xmean)/xsigma
            y = (y - ymean)/ysigma
            X_test = (X_test - xmean)/xsigma
            y_test = (y_test - xmean)/xsigma


        betas =  np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        print (betas)
        if predict == False:
            return betas
        else:
            ypred = X_test.dot(betas)
            error = (sum(np.abs(y_test - ypred)))/len(y_test)
            return error
        

class Logit():
    
    # This generates standard sygmoid function 
    @classmethod
    def sigmoid(self,matrix):
        sigmoid = 1/(1 + (np.exp(-matrix)))
        return sigmoid
    
    
    # measuring how well the algorithm performs using the loss function
    def lossfunction(self,y_train,predictions):
        lossfunc = (-y_train.T * np.log(predictions) - ((1-y_train) * np.log(1-predictions))).mean()
        return lossfunc
    
    
    def train(self,X_train,y_train,iterations = 1000,learning_rate = 0.01):
        # Generating vector with future betas [matrix w] with lenght of predictors
        betas = np.zeros(X_train.shape[1])
        for i in range(iterations):
            # create matrix Xw
            matrix = np.dot(X_train, betas)
            # create predictions
            predictions = self.sigmoid(matrix)
            # compute derivative of the loss function with respect to betas. 
            # it will show marginal change of loss function, and point out direction of minimization 
            # (since we have to minimize loss function)
            error_derivative = X_train.T.dot(predictions - y_train) / y_train.shape[0]
            # standard method from similar algorithms: etting learning rate and updating 
            # components of vector w by updating it by derivative  above , adjusted by learning rate 
            betas -= learning_rate * error_derivative
        return betas
    
    
    def predict(self,X_test,betas):
        # predictions
        matrix = np.dot(X_test, betas)
        predicted = self.sigmoid(matrix)
        return predicted
    
    def classify(self,X_test,betas, threashold = 0.5):
        matrix = np.dot(X_test, betas)
        predicted1 = self.sigmoid(matrix)
        predicted2=[]
        for obs in predicted1:
            if obs>=threashold:
                predicted2.append(1)
            else:
                predicted2.append(0)
        return predicted2
        
        














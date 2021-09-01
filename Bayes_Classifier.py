import numpy as np

# Bayesian classifier using Gaussian class conditional densities
# inspiration : https://people.eecs.berkeley.edu/~jordan/courses/281A-fall04/lectures/lec-9-30.pdf

class Bayes_classifier():
    def __init__(self, X_train,y_train):
        '''
        Constructor for Bayes classifier model
        Classifier uses linear algebra and mathematical equations to 
        compute the algorithm: no external libraries are used here.

        Parameters
        ----------
        X_train : Float values
            Data with independent variables.
        y_train : Float array
            Array with dependent variables.

        Returns
        -------
        To predict, fit prior (object.prior()), mi and sigma (object.conditional_density())
        and use them for your pluggin classifier. 
        
        Eg of use for bi-class data:
            object = Bayes_classifier(data_x_train, data_y_train)
            prior = object.prior()
            mean, variance = object.class_conditional_density()
            
            prediction = obj.plugin_classifier(data_x_predict, mean,
                                               variance, prior, 2)
            
        
        Returned are predicted classes.
        '''
        self.X_train = X_train
        self.y_train = y_train
    
    def prior(self):
        # prior is calculated with probability of one class Y being y 
        # in academic papers shows as 'pi'
        y_train = self.y_train
        deno = len(y_train)
        listtemp = np.unique(y_train)
        class_prob_vector = []
        for i in listtemp:
            num = 0
            print(i)
            for j in y_train:
                if j == i:
                    num+=1
                else:
                    continue
            prob = num/deno
            class_prob_vector.append(round(prob,5))
        return class_prob_vector
            
            
    
    def class_conditional_density(self):
        '''
        Here x is conditional of mean and covariance vectors. 
        implemented calculation from  https://people.eecs.berkeley.edu/~jordan/courses/281A-fall04/lectures/lec-9-30.pdf
        To save the mean vector and covariance matrix, call this function such
        as mean_vector, cov_matrix = object.class_conditional_density()
        '''
        X_train = self.X_train
        y_train = self.y_train
        listtemp = np.unique(y_train)
        mean_vector = np.zeros((len(listtemp),X_train.shape[1]))
        cov_matrix = np.zeros((X_train.shape[1],X_train.shape[1],len(listtemp) ))
        
        for i in listtemp:
            mean_vector[i] = np.mean(X_train[(y_train == [i])],axis=0)
            
            X_subs_mean = X_train[(y_train == i)]
            X_subs_mean = X_subs_mean - mean_vector[i]
    
            cov_matrix[:,:,i] = (np.dot(X_subs_mean.T,X_subs_mean))/len(y_train)
        return mean_vector,cov_matrix
    
    
    
    def plugin_classifier(self, X_test,mean,covariance,classprobvector,classes):
        '''
        Solving for the function and returning prediction.
        Function for the plug-in equation from the paper:
            https://people.eecs.berkeley.edu/~jordan/courses/281A-fall04/lectures/lec-9-30.pdf
            
        Parameters
        ----------
        X_test : float, values
            dependent variables you wish to predict.
        mean : 
            mean vector which was the output of class_conditional_density()
        covariance : 
            covariance matrix which was the output of class_conditional_density().
        classprobvector : 
            prior list which was the output of prior() function.
        classes : Int
            Number of classes you expect in the dataset.

        Returns
        -------
        class_prediction : List, int
            Returns predicted class, each classifing a record at index i.

        '''
        classes = np.arange(classes)
        probabilities = np.zeros((len(X_test),len(classes)))
        probabilities_norm = np.zeros((len(X_test),len(classes)))
        
        for cla in classes:
            sigma1 = np.linalg.det(covariance[:,:,cla])**-0.5
            sigma2 = np.linalg.inv(covariance[:,:,cla])
            for i in range(len(X_test)):
                xloc = X_test[i,:]
                term1  = classprobvector[cla] * sigma1
                term2 = np.exp((-0.5) * ((xloc - mean[cla]).T).dot(sigma2).dot(xloc - mean[cla]))
                prob = term1*term2
                probabilities[i,cla] = prob
        # normalisation
        for i in range(len(X_test)):
            den = float(probabilities[i,:].sum())
            probabilities_norm[i,:] = probabilities[i,:]/den
        
        class_prediction = []
        for j in range(len(probabilities_norm)):
            class_prediction.append(np.argmax(probabilities_norm[j,:]))
        return class_prediction
    










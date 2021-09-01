"""
Created on Wed Apr 14 15:57:00 2021

@author: marcinswierczewski
"""
import numpy as np
from scipy.stats import multivariate_normal

'''
K-means model
'''
def k_means(X ,clusters =  5,iterations = 20):
    '''
    Standard implementation of k-mean, using the square Euclidian distances
    to each of the centroids.
    Objective is non-convex, therefore cannot be solved analytically. 
    It uses iteration to find local optimum.
    
    For more detailed description regarding methodology 
    and equations, can visit https://statweb.stanford.edu/~tibs/stat315a/LECTURES/em.pdf

    Parameters
    ----------
    X : 
        Dataset, values.
    clusters : int
        Number of clusters. The default is 5.
    iterations : int
        Number of iterations for convergance. The default is 20.

    Returns
    -------
    assingments : 
        group that k-means assinged i-th data point to.

    '''
    # initialisation 
    assingments = np.zeros(X.shape[0])
    # mu, randomly initiated
    centroids = X[np.random.randint(0,X.shape[0],clusters)]
    choice = np.zeros(clusters)
    
    for i in range(iterations):
        # running the whole algorithm n-iteration times
        print("iteration" + str(i+1))
        # choiceprev for measuring convergence
        choiceprev = choice.copy()
        
        for j in range(len(X)):
            # updating c by difference in euclidian distances
            for c in range(clusters):
                choice[c] = np.linalg.norm(X[j] - centroids[c])
                # picking lowest value for k
            assingments[j] = np.argmin(choice)
        
        for c in range(clusters):
            # updating each centroid
            n_k = sum(assingments==c)
            if n_k !=0:
                centroids[c] = sum(X[assingments==c])/n_k
            else:
                centroids[c] = np.random.rand(len(X[1]))
        conv = (sum(choice) - sum(choiceprev))
        print("Convergence %d" % conv)
    
    return assingments
    





'''
Gaussian Mixture Model
'''
def gaussianmixture():
    return NotImplemented

def multi_gaussian(x,mean,cov):
    # creating multivariate gaussian distribution
    term1 = np.linalg.det(np.linalg.inv(cov))
    term2 = np.exp(-0.5*(x-mean).T.dot(np.linalg.inv(cov)).dot(x-mean))
    return term1*term2
    
    
def  gaussian_mixture_model(X, clusters=5,iterations=15):
    '''
    V.0.1. This is very mathematical implementation, and will be 
    optimised and improved further, in the future.
    
    It uses EM algorithm for iterations (equations can be found eg https://statweb.stanford.edu/~tibs/stat315a/LECTURES/em.pdf )

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    clusters : int
        Number of clusters the dataset will be categorised into. The default is 5.
    iterations : int
        Number of iterations for convergance. The default is 15.

    Returns
    -------
    assingments : 
        cluster that GMM assinged i-th data point to.

    '''
    # this requires further improvement
    pi = np.random.random(clusters)
    pi = np.array(list(pi[i]/sum(pi) for i in range(len(pi))))
    cov = [ np.cov(X.T) for _ in range(clusters) ]
    # for m in range(len(cov)):
    #     np.fill_diagonal(cov[m],1)
        
    # initialising phi - to be validated
    phi = np.full( shape=X.shape, fill_value=1/clusters)
    # means generated with positive integers of existing rows to avoid non singular matrices with some datasets
    random_row = np.random.randint(low=0, high=X.shape[0], size=clusters)
    mean = [  X[row_index,:] for row_index in random_row ]
    # cov = [ np.cov(X.T) for _ in range(clusters) ]
    
    for i in range(iterations):
        print("iter" + str(i))
        phi = []
        for j in range((X.shape[0])):
            num = []
            for k in range(clusters):
                    # creating multivariate gaussian distribution on xi matrix of x's
                    gauss = multi_gaussian(X[j],mean[k],cov[k])
                    num.append(pi[k]*gauss)
                #   alternative /validation: 
                    # distribution = multivariate_normal(
                    # mean=mean[k], 
                    # cov=cov[k])
                    # num.append(distribution.pdf(X[j]))
            deno = sum(num)
            phi.append([x / deno for x in num])
        phi = np.array(phi)
        # check source of those nans
        
        phi = np.nan_to_num(phi)
        phi = np.array(phi)
        pi = (np.sum(phi,axis=0)) / (X.shape[0])
        for k in range(clusters):
            # mean[k] = phi[:,k].T.dot(X) / (np.sum(phi,axis=0))[k]
            mean[k] = (X * phi[:,[k]]).sum(axis=0) / (phi[:,[k]].sum())
        for k in range(clusters):
            # calculating cov using updated value for mean
            # xi = X - mean[k]
            # num = xi.T.dot(np.diag(phi[:,k])).dot(xi)
            weight = phi[:, [k]]
            total_weight = weight.sum()
            cov[k] = np.cov(X.T, 
                aweights=(weight/total_weight).flatten(), 
                bias=True)
    
    assignment = []
    for p in range(len(phi)):
        assignment.append(np.argmax(phi[p]))
    assignment=  np.array(assignment)
    return (assignment)






















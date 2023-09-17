# -*- coding: utf-8 -*-
"""
@author: KALYANI BURANDE
DATE: 02/03/2022
COURSE: CS5691 PRML
ASSIGNMENT 1 : PCA, KERNEL PCA, K-MEANS, SPECTRAL CLUSTERING

PROBLEM STATEMENT:
iii. Write a piece of code to implement the Kernel PCA algorithm on this dataset.
Use the following kernels :
A. polynomial kernel for d = {2,3}
B. Radial Basis Function for sigma ={0.1, 0.2, 0.3, ..1}

Plot the projection of each point in the dataset onto the top-2 components for
each kernel. Use one plot for each kernel and in the case of (B), use a dierent
plot for each value of sigma.
"""

print("Kernel PCA on dataset for Polynomial and Radial Basis Kernels --> \n")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#read dataset from csv file in to a pandas dataframe
data = pd.read_csv("Dataset.csv", header = None)

#Dataset exploration using pd
# print(data.head())
# print(data.iloc[0:5,0])
# print(data.iloc[0:5,1])
# print(data.iloc[:,0])
# print(data.iloc[:,1])

#Compute Covariance matrix using pd
# cov_matrix = data.cov(ddof = 0)
# print(cov_matrix)

#Compute Covariance matrix using numpy
x1 = np.array(data.iloc[:,0])
x2 = np.array(data.iloc[:,1])
x1_len = len(x1)
#print(x1, x2)
# print("x1 shape", x1.shape)
X = np.column_stack([x1, x2])
# print("Dataset in R^nxd:\n", X)#x is nxd mat
#X_prime = X.T 
#print("Dataset in R^dxn:\n", X_prime)#x is dxn mat, use this fro kernel PCA
#print("mean of dataset =", X.mean(axis=0))

select_kernel = 0
# K = np.zeros((x1_len,x1_len))

def Compute_Polynomial_Kernel_Output(select_kernel):

    if select_kernel == 2:
        #compute kernel matrix output
        #k(x,y) = (1 + x.T y)^d ; d = 2
        Kernel_matrix = np.square((1 + np.dot(X,X.T)))
        #print("K\n:", K)
        #print(K.shape)#1000x1000
        #row,col = K.shape
        
    if select_kernel == 3:
        #compute kernel matrix output
        #k(x,y) = (1 + x.T y)^d ; d = 3
        Kernel_matrix = np.power((1 + np.dot(X,X.T)),3)
        #print("K\n:", K)
        #print(K.shape)#1000x1000
        #row,col = K.shape
    
    return Kernel_matrix


def Compute_Exponential_Kernel_Output(sigma):
    
    rows, cols = (x1_len, x1_len)
    Kernel_matrix = [[0]*cols]*rows
    # print(len(Kernel_matrix))
    # print(len(Kernel_matrix[0]))
    
    #compute kernel matrix output fro radial basis function for passed sigma value
    for i in range(len(x1)):
        for j in range(len(x2)):
            distance_btwn_points = math.pow((x1[i]-x2[j]),2)
            exponent = distance_btwn_points * (-1.0 / (2.0 * (sigma *sigma)))
            Kernel_matrix[i][j] = math.exp(exponent)
    
    return Kernel_matrix  


def Get_Aplha_vector_and_Datapoints_Projection_on_Alpha(K):

    #get modified Kernel matrix by centering dataset in featurespace 
    K_rows = K.shape[0] 
    one_K_rows = np.ones((K_rows,K_rows)) / K_rows
    K = K - one_K_rows.dot(K) - K.dot(one_K_rows) + one_K_rows.dot(K).dot(one_K_rows) 
    
    
    eig_value, eig_vector = np.linalg.eig(K)
    # print("Eigen values are:\n", eig_value, "\n")
    # print("Eigen vectors are (read columnwise):\n", eig_vector, "\n")
    
    
    #get indices of first 2 eigen values with max values
    no_of_eigen_values = 2
    eig_value_indices = (-eig_value).argsort()[:no_of_eigen_values]
    # print(eig_value_indices)
    # print(eig_value_indices[0], eig_value_indices[1])
     
    
    #get the eigen vector corresponding to max eigen value of K
    max_eig_value_index_1 = eig_value_indices[0]
    # print("index of max eigen value :\n", max_eig_value_index_1)
   
    # #normalise eigen vectors of K to get alpha_vector
    alpha_1 = eig_vector[:,max_eig_value_index_1] / (np.sqrt(np.absolute(eig_value[max_eig_value_index_1])))
    # print("eigen vector with max eigen value: \n", alpha_1)
    # print("shape of alpha_1:", alpha_1.shape)
    
    #get the eigen vector corresponding to secong max eigen value of K
    max_eig_value_index_2 = eig_value_indices[1]
    # print("index of second max eigen value :\n", max_eig_value_index_2)
    
    # #normalise eigen vectors of K to get alpha_vector
    alpha_2 = eig_vector[:,max_eig_value_index_2] / (np.sqrt(np.absolute(eig_value[max_eig_value_index_2])))
    # print("eigen vector with second max eigen value: \n", alpha_2)
    # print("shape of alpha_2:", alpha_2.shape)
     
    
    #projection of all points on alpha_1 vector
    # print("shape of K.T:\n", K.T.shape)
    # print("shape of alpha_1:\n", alpha_1.shape)
    projection_alpha_1 = np.dot(K.T, alpha_1)
    # print("Projections of all points on alpha_1 vector:\n", projection_alpha_1)
    #print("shape of projection_alpha_1:\n", projection_alpha_1.shape)
    
    
    #projection of all points on alpha_2 vector
    # print("shape of K.T:\n", K.T.shape)
    # print("shape of alpha_2:\n", alpha_2.shape)
    projection_alpha_2 = np.dot(K.T, alpha_2)
    # print("Projections of all points on alpha_2 vector:\n", projection_alpha_2)
    #print("shape of projection_alpha_2:\n", projection_alpha_2.shape)
    
    return alpha_1, projection_alpha_1, alpha_2, projection_alpha_2

 
def Plot_Datapoint_Projections_for_Kernel(alpha_vector_1, projection_alpha_1, alpha_vector_2, projection_alpha_2, sigma = None):
    
    #display scatter plot for all kernels 
    #sigma = None: scatter plot for polynomial kernel
    if sigma == 2 or sigma == 3:
        plt.scatter(projection_alpha_1, alpha_1, color = 'r', marker = ".", label = "alpha_vector_1")
        plt.scatter(projection_alpha_2, alpha_2, color = 'g', marker = ".", label = "alpha_vector_2")#, linewidths = 0.3)
        plt.title("Projection of datapoints onto top-2 alpha_vectors of kernel"+ " d: " +str(sigma))
    
    #sigma = passed valued: scatter plot for RBF kernel with passed sigma value
    else:
        plt.scatter(projection_alpha_1, alpha_1, color = 'r', marker = ".", label = "alpha_vector_1")
        plt.scatter(projection_alpha_2, alpha_2, color = 'g', marker = ".", label = "alpha_vector_2")#, linewidths = 0.3)
        title = "Projection of datapoints onto top-2 alpha_vectors of kernel " + "sigma:" +str(round(sigma, 2))
        plt.title(title)
    
    
    plt.xlabel("alpha_vector_components")
    plt.ylabel("Projection of datapoint on alpha_vector")
    # plt.title("Projection of datapoints onto top-2 alpha_vectors of kernel")
    plt.legend()
    plt.show() 
    

#plot scatter plot for polynomial kernel with d = 2
K = Compute_Polynomial_Kernel_Output(2)   
alpha_1, projection_alpha_1, alpha_2, projection_alpha_2 = Get_Aplha_vector_and_Datapoints_Projection_on_Alpha(K)
Plot_Datapoint_Projections_for_Kernel(alpha_1, projection_alpha_1, alpha_2, projection_alpha_2, 2)

#plot scatter plot for polynomial kernel with d = 3
K = Compute_Polynomial_Kernel_Output(3)   
alpha_1, projection_alpha_1, alpha_2, projection_alpha_2 = Get_Aplha_vector_and_Datapoints_Projection_on_Alpha(K)
Plot_Datapoint_Projections_for_Kernel(alpha_1, projection_alpha_1, alpha_2, projection_alpha_2, 3)

#plot scatter plot for RBF kernel with sigma values 
sigma_range = np.arange(0.1, 1.1, 0.1)
sigma_range_listing = list(sigma_range)
for sigma in sigma_range_listing:
    K = Compute_Exponential_Kernel_Output(sigma)
    # print(len(K))
    # print(len(K[0]))
    K = np.array(K)
    # print(K.shape)
    alpha_1, projection_alpha_1, alpha_2, projection_alpha_2 = Get_Aplha_vector_and_Datapoints_Projection_on_Alpha(K)
    Plot_Datapoint_Projections_for_Kernel(alpha_1, projection_alpha_1, alpha_2, projection_alpha_2, sigma)
 
print("Projections of all datapoints on top 2 components of each kernel is plotted on scatter plots.")

"""
Result:

Kernel PCA on dataset for Polynomial and Radial Basis Kernels --> 

Projections of all datapoints on top 2 components of each kernel is plotted on scatter plots.
"""








# -*- coding: utf-8 -*-
"""
@author: KALYANI BURANDE
DATE: 02/03/2022
COURSE: CS5691 PRML
ASSIGNMENT 1 : PCA, KERNEL PCA, K-MEANS, SPECTRAL CLUSTERING

PROBLEM STATEMENT:
(1) You are given a data-set with 1000 data points each in R2.
i. Write a piece of code to run the PCA algorithm on this data-set. 
How much of the variance in the data-set is explained by each of the principal components?
"""

print("PCA with centering dataset--> \n")

import numpy as np
import pandas as pd

#read data from csv file 
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
#print(x1, x2) 
X = np.column_stack([x1, x2])
# print("mean of dataset =", X.mean(axis=0))
X = X - X.mean(axis=0) 
covariance_matrix_of_X = np.dot(X.T, X) / len(x1)
print("Covariance matrix of X:\n", covariance_matrix_of_X, "\n")

#This does not work 
# x1 = x1 - x1.mean()
# x2 = x2 - x2.mean()
# manual_cov = np.dot(x1.T, x2) / len(x1)
# print(manual_cov)

#Obtain eigenvalues and eigenvectors (eigenvectors arranged in columns) of covariance matrix x
eig_value, eig_vector = np.linalg.eig(covariance_matrix_of_X)
# print("Eigen values are:\n", eig_value, "\n")
# print("Eigen vectors are (read columnwise):\n", eig_vector, "\n")

#Get and print eigen vectors 
#eig_vector_1 = np.array([eig_vector[0,0], eig_vector[1,0]])
eig_vector_1 = eig_vector[:,0]
# print(eig_vector_1)

# eig_vector_2 = np.array([eig_vector[0,1], eig_vector[1,1]])
eig_vector_2 = eig_vector[:,1]
# print(eig_vector_2)

print("Printing Eigen vectors and Eigen values of covariance matrix-->\n")
print("Eigen vector with largest eigen value or First Principal Component:\n", "w1 = ", eig_vector_2, "with eigen value lambda 1 =", eig_value[1], "\n")
print("Eigen vector with second largest eigen value or Second Principal Component:\n", "w2 = ", eig_vector_1, "with eigen value lambda 2 =", eig_value[0], "\n")


#w1 and w2 are orthonormal vectors 
# print("Norm of w1 =", np.linalg.norm(eig_vector_1))
# print("Norm of w2 =", np.linalg.norm(eig_vector_2))
# print("Dot product of Eigen vectors of covariance matrix is: ", np.dot(eig_vector_1, eig_vector_2), "\n")

#Compute Variance explained by each principal component
var_1 = 0
for i in range(0, len(X)):
    var_1 = var_1 + (np.square(np.dot(X[i].transpose(), eig_vector_2)))
    
var_1 = var_1 / len(X)

var_2 = 0
for j in range(0, len(X)):
    var_2 = var_2 + (np.square(np.dot(X[j].transpose(), eig_vector_1)))
    
var_2 = var_2 / len(X)

print("Amount of variance in dataset explained by w1 = ", var_1/(var_1+var_2)*100, " %")
print("Amount of variance in dataset explained by w2 = ", var_2/(var_1+var_2)*100, " %")
print("Total variance explained by w1 and w2 in %:", var_1/(var_1+var_2)*100 + var_2/(var_1+var_2)*100) #prints 100 

"""
Result:

PCA without centering dataset--> 

Mean of dataset = [4.075e-07 2.227e-07]
Covariance matrix:
 [[14.76615576  0.80885904]
 [ 0.80885904 16.85536339]] 

Printing Eigen vectors and Eigen values of covariance matrix-->

Eigen vector with largest eigen value or First Principal Component:
 w1 =  [-0.323516  -0.9462227] with eigen value lambda 1 = 17.131914402444487 

Eigen vector with second largest eigen value or Second Principal Component:
 w2 =  [-0.9462227  0.323516 ] with eigen value lambda 2 = 14.489604749330738 

Amount of variance in dataset explained by w1 =  54.1780245288522  %
Amount of variance in dataset explained by w2 =  45.8219754711478  %
Total variance explained by w1 and w2 in %: 100.0
"""







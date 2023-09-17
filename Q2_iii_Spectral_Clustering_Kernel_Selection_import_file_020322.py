"""
@author: KALYANI BURANDE
DATE: 02/03/2022
COURSE: CS5691 PRML
ASSIGNMENT 1 : PCA, KERNEL PCA, K-MEANS, SPECTRAL CLUSTERING

PROBLEM STATEMENT:
(2)You are given a data-set with 1000 data points each in R2.
iii. Run the spectral clustering algorithm (spectral relaxation of K-means using Kernel PCA) k = 4. 
Choose an appropriate kernel for this data-set and plot the clusters obtained in different colors. 
Explain your choice of kernel based on the output you obtain.
"""

"""
This is a file imported as module in Q2_iii_Spectral_Clustering_Kernel_Selection_MAIN_file_020322.py file
Please keep this file in same directory as the Q2_iii_Spectral_Clustering_Kernel_Selection_MAIN_file_020322.py file
Please run Q2_iii_Spectral_Clustering_Kernel_Selection_MAIN_file_020322.py to invoke this file as module 
"""

print("This is a file imported as module in Q2_iii_Spectral_Clustering_Kernel_Selection_MAIN_file_020322.py file\n")
print("Please keep this file in same directory as the Q2_iii_Spectral_Clustering_Kernel_Selection_MAIN_file_020322.py file\n")
print("Please run Q2_iii_Spectral_Clustering_Kernel_Selection_MAIN_file_020322.py to invoke this file as module\n")


#COMPUTING RBF KERNEL MATRIX FOR SPECTRAL CLUSTERING PROBLEM IN PRML ASSIGN. 1 Q2.iii
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

data = pd.read_csv("Dataset.csv", header = None)

#Dataset exploration using pd
# print(data.head())
# print(data.iloc[0:5,0])
# print(data.iloc[0:5,1])
# print(data.iloc[:,0])
# print(data.iloc[:,1])

x1 = np.array(data.iloc[:,0])
x2 = np.array(data.iloc[:,1])
x1_len = len(x1)
#print(x1, x2)
# print("x1 shape", x1.shape)
X = np.column_stack([x1, x2])
# print("Dataset in R^nxd:\n", X)#x is nxd mat
#X_prime = X.T 
#print("Dataset in R^dxn:\n", X_prime)#x is dxn mat, use this fro kernel PCA
# print("mean of dataset =", X.mean(axis=0))

select_kernel = 0
# K = np.zeros((x1_len,x1_len))

no_of_clusters = 4 
eig_value_rows = 0
H_star_unit_row_vector_normalied = [[]*no_of_clusters]*eig_value_rows 

def Compute_Polynomial_Kernel_Output(select_kernel):

    if select_kernel == 2:
        #compute kernel matrix output
        #k(x,y) = (1 + x.T y)^d ; d = {2, 3}
        K = np.square((1 + np.dot(X,X.T)))
        #print("K\n:", K)
        #print(K.shape)#1000x1000
        #row,col = K.shape
        
    if select_kernel == 3:
        #compute kernel matrix output
        #k(x,y) = (1 + x.T y)^d ; d = {2, 3}
        K = np.power((1 + np.dot(X,X.T)),3)
        #print("K\n:", K)
        #print(K.shape)#1000x1000
        #row,col = K.shape
    
    return K


def Compute_Exponential_Kernel_Output(sigma):
        
    rows, cols = (x1_len, x1_len)
    K = [[0]*cols]*rows
    # print(len(K))
    # print(len(K[0]))

    for i in range(len(x1)):
        for j in range(len(x2)):
            distance_btwn_points = math.pow((x1[i]-x2[j]),2)
            exponent = distance_btwn_points * (-1.0 / (2.0 * (sigma *sigma)))
            K[i][j] = math.exp(exponent)
            
    return K  


def Get_Aplha_vector_and_Datapoints_Projection_on_Alpha(K, no_of_clusters):

    #get modified K matrix by centering dataset in featurespace 
    # K_rows = K.shape[0] 
    # one_K_rows = np.ones((K_rows,K_rows)) / K_rows
    # K = K - one_K_rows.dot(K) - K.dot(one_K_rows) + one_K_rows.dot(K).dot(one_K_rows) 
    
    
    eig_value, eig_vector = np.linalg.eig(K) 
    # print("Eigen values are:\n", eig_value, "\n")
    # print("Eigen vectors are (read columnwise):\n", eig_vector, "\n")
    # print("Eigen vectors no. of columns :\n", len(eig_vector), "\n")
    # print("Eigen vectors no. of rows :\n", len(eig_vector[0]), "\n")
    
    #get indices of first 2 eigen values with max values
    no_of_eigen_values = no_of_clusters
    eig_value_indices = (-eig_value).argsort()[:no_of_eigen_values]
    # eig_value_indices_sorted = np.sort(no_of_eigen_values)
    # print(eig_value_indices)
    eig_value_indices = eig_value_indices.tolist()
    # print(type(eig_value_indices))
    eig_value_indices_sorted = sorted(eig_value_indices)
    # print("eig_value_indices_sorted:", eig_value_indices_sorted)
    # print(eig_value_indices[0], eig_value_indices[1])
    
    eig_value_rows =eig_value.shape[0] 
    H_star = [[]*no_of_clusters]*eig_value_rows
    H_star = np.array(H_star)
    # print("H_star shape at creation:", H_star.shape)
    
    for index in eig_value_indices_sorted:
        # print("current index or column of H_star:", index)
        # print("top k=4 eigen values:", eig_value[index])
        # print("top k=4 eigen vectors:", eig_vector[:, index], "\n\n")
        # print("top k=4 eigen vectors shape:", eig_vector[:, index].shape, "\n\n")
        #got all vectors append them to form H matrix
        H_star = np.column_stack((H_star, eig_vector[:, index]))
    
    #print details of H_star (proxy matrix for ZxL^1/2)
    # print("H_star shape:", H_star.shape)
    # print("H_star shape:", type(H_star))
    # print("H_star row1 :", H_star[0:2,0:4])
    
    #now normalise all 'N=1000=no. of samples' rows of H_star
    H_star_unit_row_vector_normalied = H_star / np.linalg.norm(H_star, axis=-1)[:, np.newaxis]
    # H_star_unit_row_vector_normalied = H_star
    # print("H_star_unit_row_vector_normalied shape:", H_star_unit_row_vector_normalied.shape)
    # print("H_star_unit_row_vector_normalied TYPE:", type(H_star_unit_row_vector_normalied))
    # print("H_star_unit_row_vector_normalied row1 :", H_star_unit_row_vector_normalied[0:2,0:4])
    
    #get the eigen vector corresponding to max eigen value of K
    # max_eig_value_index_1 = eig_value_indices[0]
    # print("index of max eigen value :\n", max_eig_value_index_1)
    # alpha_1 = eig_vector[:,max_eig_value_index_1] / (np.sqrt(np.absolute(eig_value[max_eig_value_index_1])))
    # print("eigen vector with max eigen value: \n", alpha_1)
    # print("shape of alpha_1:", alpha_1.shape)
    
    #get the eigen vector corresponding to secong max eigen value of K
    # max_eig_value_index_2 = eig_value_indices[1]
    # print("index of second max eigen value :\n", max_eig_value_index_2)
    # alpha_2 = eig_vector[:,max_eig_value_index_2] / (np.sqrt(np.absolute(eig_value[max_eig_value_index_2])))
    # print("eigen vector with second max eigen value: \n", alpha_2)
    # print("shape of alpha_2:", alpha_2.shape)
     
    
    # #normalise eigen vectors of K to get alpha_vector
    # eig_value_absolute = np.absolute(eig_value)
    # eig_value_sqrt = np.sqrt(eig_value_absolute)
    # print("eig_value_absolute:\n", eig_value_sqrt)
    # for i in range(0, row):
    #     alpha_vector = eig_vector[:,i] / eig_value_sqrt[i]
    # print("alpha_vector: \n", alpha_vector) 
    
    
    #projection of all points on alpha_1 vector
    # print("shape of K.T:\n", K.T.shape)
    # print("shape of alpha_1:\n", alpha_1.shape)
    # projection_alpha_1 = np.dot(K.T, alpha_1)
    # print("Projections of all points on alpha_1 vector:\n", projection_alpha_1)
    #print("shape of projection_alpha_1:\n", projection_alpha_1.shape)
    
    
    
    #projection of all points on alpha_2 vector
    # print("shape of K.T:\n", K.T.shape)
    # print("shape of alpha_2:\n", alpha_2.shape)
    # projection_alpha_2 = np.dot(K.T, alpha_2)
    # print("Projections of all points on alpha_2 vector:\n", projection_alpha_2)
    #print("shape of projection_alpha_2:\n", projection_alpha_2.shape)
    # return alpha_1, projection_alpha_1, alpha_2, projection_alpha_2
    return H_star_unit_row_vector_normalied


"""
def Plot_Datapoint_Projections_for_Kernel(alpha_vector_1, projection_alpha_1, alpha_vector_2, projection_alpha_2, m = None):
    if m == None:
        plt.scatter(projection_alpha_1, alpha_1, color = 'r', marker = ".", label = "alpha_vector_1")
        plt.scatter(projection_alpha_2, alpha_2, color = 'g', marker = ".", label = "alpha_vector_2")#, linewidths = 0.3)
        plt.title("Projection of datapoints onto top-2 alpha_vectors of kernel")
    else:
        plt.scatter(projection_alpha_1, alpha_1, color = 'r', marker = ".", label = "alpha_vector_1")
        plt.scatter(projection_alpha_2, alpha_2, color = 'g', marker = ".", label = "alpha_vector_2")#, linewidths = 0.3)
        title = "Projection of datapoints onto top-2 alpha_vectors of kernel " + "sigma:" +str(round(m, 2))
        plt.title(title)
    
    
    plt.xlabel("alpha_vector_components")
    plt.ylabel("Projection of datapoint on alpha_vector")
    # plt.title("Projection of datapoints onto top-2 alpha_vectors of kernel")
    plt.legend()
    plt.show() 
    

# K = Compute_Polynomial_Kernel_Output(0)   
# alpha_1, projection_alpha_1, alpha_2, projection_alpha_2 = Get_Aplha_vector_and_Datapoints_Projection_on_Alpha(K)
# Plot_Datapoint_Projections_for_Kernel(alpha_1, projection_alpha_1, alpha_2, projection_alpha_2)

# K = Compute_Polynomial_Kernel_Output(1)   
# alpha_1, projection_alpha_1, alpha_2, projection_alpha_2 = Get_Aplha_vector_and_Datapoints_Projection_on_Alpha(K)
# Plot_Datapoint_Projections_for_Kernel(alpha_1, projection_alpha_1, alpha_2, projection_alpha_2)
"""

def Compute_H_star_matrix(select_a_kernel):
    # compute normalised kerenel matrix as per kernel function selection from main file and return normalised
    # H matrix to main module 
    
    sigma_range = np.arange(0.8, 0.9, 0.1)
    sigma_range_listing = list(sigma_range)
    
    if select_a_kernel == 0:
        #get kernel matrix for polynomial kernel with d = 2
        K = Compute_Polynomial_Kernel_Output(2)      
    
    if select_a_kernel == 1:
        #get kernel matrix for polynomial kernel with d = 3
        K = Compute_Polynomial_Kernel_Output(3)
   
    if select_a_kernel == 2:
        #get RBF kernel matrix with sigma = 0.8
        for m in sigma_range_listing:
            K = Compute_Exponential_Kernel_Output(m)

    # Plot_Datapoint_Projections_for_Kernel(alpha_1, projection_alpha_1, alpha_2, projection_alpha_2, m)
        
    # print("no. of Columns of K:", len(K))
    # print("no. of Rows of K:", len(K[0]))
    K = np.array(K)
    # print(K.shape)
    H_star_unit_row_vector_normalied = Get_Aplha_vector_and_Datapoints_Projection_on_Alpha(K, no_of_clusters)
    
    
    return H_star_unit_row_vector_normalied


 









#this does not work
# #get next largest eigen value and corresponding eigen vector 
# reduced_eig_value = np.delete(eig_value, max_eig_value_index_1)
# max_eig_value_index_2 = np.argmax(reduced_eig_value)
# print("index of second max eigen value :\n", max_eig_value_index_2)
# reduced_eig_vector = np.delete(eig_vector, max_eig_value_index_1)
# alpha_2 = reduced_eig_vector[max_eig_value_index_2]
# print("eigen vector with second max eigen value: \n", alpha_2)
# print("shape of alpha_2:", alpha_2.shape)


"""
for i in range(0, len(X)):
    var_1 = var_1 + (np.square(np.dot(X[i].transpose(), eig_vector_2)))
    
var_1 = var_1 / len(X)



X = X - X.mean(axis=0) 
manual_cov = np.dot(X.T, X) / len(x1)
print("Covariance matrix:\n", manual_cov, "\n")
"""
"""
#This does not work 
# x1 = x1 - x1.mean()
# x2 = x2 - x2.mean()
# manual_cov = np.dot(x1.T, x2) / len(x1)
# print(manual_cov)

#Obtain eigenvalues and eigenvectors (eigenvectors arranged in columns) 
#of covariance matrix 
eig_value, eig_vector = np.linalg.eig(manual_cov)
print("Eigen values are:\n", eig_value, "\n")
print("Eigen vectors are (read columnwise):\n", eig_vector, "\n")

#Get and print eigen vectors 
#eig_vector_1 = np.array([eig_vector[0,0], eig_vector[1,0]])
eig_vector_1 = eig_vector[:,0]
# print(eig_vector_1)

# eig_vector_2 = np.array([eig_vector[0,1], eig_vector[1,1]])
eig_vector_2 = eig_vector[:,1]
# print(eig_vector_2)

print("Eigen vector of covariance matrix-->")
print("Eigen vector with largest eigen value or First Principal Component:\n", "w1 = ", eig_vector_2, "with eigen value lambda 1 =", eig_value[1], "\n")
print("Eigen vector with second largest eigen value or Second Principal Component:\n", "w2 = ", eig_vector_1, "with eigen value lambda 2 =", eig_value[0], "\n")

#w1 and w2 are orthonormal vectors 
print("Norm of w1 =", np.linalg.norm(eig_vector_1))
print("Norm of w2 =", np.linalg.norm(eig_vector_2))
print("Dot product of Eigen vectors of covariance matrix is: ", np.dot(eig_vector_1, eig_vector_2), "\n")

#Variance explained by each principal component
var_1 = 0
for i in range(0, len(X)):
    var_1 = var_1 + (np.square(np.dot(X[i].transpose(), eig_vector_2)))
    
var_1 = var_1 / len(X)

var_2 = 0
for j in range(0, len(X)):
    var_2 = var_2 + (np.square(np.dot(X[j].transpose(), eig_vector_1)))
    
var_2 = var_2 / len(X)

print("Amount of variance in dataset explained by w1 = ", var_1/(var_1+var_2), " %")
print("Amount of variance in dataset explained by w2 = ", var_2/(var_1+var_2), " %")

"""






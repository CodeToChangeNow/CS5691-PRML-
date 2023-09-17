"""
@author: KALYANI BURANDE
DATE: 02/03/2022
COURSE: CS5691 PRML
ASSIGNMENT 1 : PCA, KERNEL PCA, K-MEANS, SPECTRAL CLUSTERING

PROBLEM STATEMENT:
(2)You are given a data-set with 1000 data points each in R2.
iv. Instead of using the method suggested by spectrl clustering to map eigen vectors to cluster 
assignments, use the following method: Assign data point i to cluster l whenever 
l = arg max vij for j = 1,...,k
where vij belongs to R^n is the eigenvector of the kernel matrix associated with the jth largets eigenvalue.
How does this mapping perform for this dataset ? Explain your insights.
"""

"""MAIN FILE
Q2_iv_Spectral_Clustering_ARG_MAX_import_file_020322.py file is imported into this MAIN FILE

Run this main file and Q2_iv_Spectral_Clustering_ARG_MAX_import_file_020322 module will be 
automatically invoked 

"""


print("Plotting coloured clusters (k=4) for fixed rand. init. for ARG MAX based cluster assignment and ARG MAX assignment checking performance")

import numpy as np
import pandas as pd
import random, math 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from Q2_iv_Spectral_Clustering_ARG_MAX_import_file_020322 import Compute_H_star_matrix #, H_star_unit_row_vector_normalied
# H_star_unit_row_vector_normalied = Compute_H_star_matrix()
# print("in main filr H_star_unit_row_vector_normalied:", H_star_unit_row_vector_normalied)
# print("H_star_unit_row_vector_normalied shape MAIN:", H_star_unit_row_vector_normalied.shape)
# print("H_star_unit_row_vector_normalied TYPE:", type(H_star_unit_row_vector_normalied))
# print("H_star_unit_row_vector_normalied row1 MAIN:", H_star_unit_row_vector_normalied[0:2,0:4])
# random.seed(20)
# print("is complex?",   np.iscomplex(H_star_unit_row_vector_normalied[:,:]) )


#Initialise variables 
k_means_count = 4 # initial no. of k means classes 
final_k_with_centroid = {} #dict to hold k:centroid
final_k_with_cost = {}     #dict to hold k:cost
J_list = []                #list for computed cost values per k 


def Get_K_Means_Centroid_Cordinates(k, H_star_unit_row_vector_normalied):
    
        #compute cost for each kto find optimum k and its coordinates
        x1 = np.absolute(H_star_unit_row_vector_normalied[:,0])
        x2 = np.absolute(H_star_unit_row_vector_normalied[:,1])
        x3 = np.absolute(H_star_unit_row_vector_normalied[:,2])
        x4 = np.absolute(H_star_unit_row_vector_normalied[:,3])
        X = np.column_stack([x1, x2, x3, x4])

        x1_data = x1.tolist()
        x2_data = x2.tolist()
        x3_data = x3.tolist()
        x4_data = x4.tolist()

        # print("x1_data:", x1_data)

        # m = number of sample points available in ground truth data
        samples_count = len(x1_data) 
        # print("number of samples:", samples_count)
        
        
        k_means_count = k
        # print("Number of classes = K = " + str(k_means_count))
        k_means_centroid_cordinates = [] #stores all coordinates for all k in a list
        k_means_class = []#store current k class labels in a list
        
        #craete k class labels  
        for k in range(0, k_means_count):
            k_means_class.append(str(k))   
        # print("K-means class labels are:", k_means_class)    
        
        assigned_class = []
        sample_class_dict = {}#create dict, x1:assigned class
        sample_class_dict_2 = {}#create dict, x2:assigned class
        sample_class_dict_3 = {}#create dict, x3:assigned class
        sample_class_dict_4 = {}#create dict, x4:assigned class
        
        #for every sample assign cluster no. = column index of element with larget value 
        for sample in range(0, samples_count):
            # print("X row 1:", X[0,:])
            # print("X row 1:", max(X[0,:]))
            max_index = np.argmax(X[sample,:])
            # print("index of max element in X row 1:", max_index)
            cluster_no = str(max_index)
            # print("index of max element in X row 1 = cluster no.:", cluster_no)
            assigned_class.append(cluster_no)
            sample_class_dict[x1_data[sample]] =k_means_class[max_index]
            sample_class_dict_2[x2_data[sample]] =k_means_class[max_index]
            sample_class_dict_3[x3_data[sample]] =k_means_class[max_index]
            sample_class_dict_4[x4_data[sample]] =k_means_class[max_index]
            
        # print("assigned class:", assigned_class)
        """
        #generate random centroid coordinates
        temp = [] #holds centroid values per k 
        for ele in range (0,k_means_count):
            centroid_x1 = random.uniform(min(x1_data),max(x1_data))
            centroid_x2 = random.uniform(min(x2_data),max(x2_data))
            centroid_x3 = random.uniform(min(x3_data),max(x3_data))
            centroid_x4 = random.uniform(min(x4_data),max(x4_data))
            temp.append(centroid_x1)
            temp.append(centroid_x2)
            temp.append(centroid_x3)
            temp.append(centroid_x4)
            k_means_centroid_cordinates.append(temp)
            temp = []
        
        # print("\nk_means_centroid_cordinates random intiialisations at start:\n", k_means_centroid_cordinates)  
        # print("centroid x1:", k_means_centroid_cordinates[0][0])  
        # print("centroid x2:", k_means_centroid_cordinates[0][1])  
        # print("centroid x3:", k_means_centroid_cordinates[1][0])  
        # print("centroid x4:", k_means_centroid_cordinates[1][1])  
        
        #search coordinate for current k 
        iterations = 100
        iter_no = 0
        
        while(iter_no < iterations):
            distance_from_centroid_each_sample = []
            assigned_class = []
            sample_class_dict = {}#create dict, x1:assigned class
            sample_class_dict_2 = {}#create dict, x2:assigned class
            sample_class_dict_3 = {}#create dict, x3:assigned class
            sample_class_dict_4 = {}#create dict, x4:assigned class
            
            #assign nearest class label to each sample
            for sample in range(0, samples_count):
                # print("sample no:", sample)
                #for every sample do this
                for centroids in k_means_centroid_cordinates:       
                    # print("centroids:", centroids)
                    # print(centroids[0])
                    # print(centroids[1])
                    #compute distance of label to each centroids
                    distance_1 = pow((x1_data[sample] - centroids[0]),2)
                    distance_2 = pow((x2_data[sample] - centroids[1]),2)
                    distance_3 = pow((x3_data[sample] - centroids[2]),2)
                    distance_4 = pow((x4_data[sample] - centroids[3]),2)
                    # print("distance from centroids:", sample, distance_1, distance_2, distance_3, distance_4)
                    dist = pow((distance_1 + distance_2 + distance_3 + distance_4),0.5)
                    # print("pow distance from centroids:", sample, dist)
                    distance_from_centroid_each_sample.append(dist)
                # print("PPPPPPdistance_from_centroid_each_sample:", distance_from_centroid_each_sample)
                min_distance = min(distance_from_centroid_each_sample)
                min_index = distance_from_centroid_each_sample.index(min_distance)
                # print("PPPmin_index:", sample, min_index)
                #print("value at min_index:", k_means_class[min_index], "\n")
                
                #assign class to sample corres. to centroid with smallest distance
                sample_class_dict[x1_data[sample]] =k_means_class[min_index]
                sample_class_dict_2[x2_data[sample]] =k_means_class[min_index]
                sample_class_dict_3[x3_data[sample]] =k_means_class[min_index]
                sample_class_dict_4[x4_data[sample]] =k_means_class[min_index]
                
                assigned_class.append(k_means_class[min_index])
                distance_from_centroid_each_sample = []
            
        #     print("sample_class_dict count:", len(sample_class_dict))
        #     print("sample_class_dict:", sample_class_dict)
        #     print("sample_class_dict_2 count:", len(sample_class_dict_2))
        #     print("sample_class_dict_3 count:", len(sample_class_dict_3))
        #     print("sample_class_dict_4 count:", len(sample_class_dict_4)  )
        #     print(sample_class_dict_2) 
        #     print("len of assigned_class:", len(assigned_class) )  
        #     print("count of datapoints with cluster reassignemnt:",len(sample_class_dict))# + len(sample_class_dict_2) +len(sample_class_dict_3) +len(sample_class_dict_4) )
            
            #craete new list to store updated class coordinates
            k_means_centroid_cordinates_new = []
            
            #temporary lists for computing new class coordiantes
            temp_centroid_x1 = []
            temp_centroid_x2 = []    
            temp_centroid_x3 = []
            temp_centroid_x4 = []  
            temp = []                    
            for c in k_means_class:#start with every class
                #club x1 samples with same assigned class labels 
                for k,v in sample_class_dict.items():
                    if v == c:
                        temp_centroid_x1.append(k)
                    else:
                        continue
                # print("c PRINTING LENtemp_centroid_x1:", c, len(temp_centroid_x1))
                #update x1 part of centroid

                new_centroid_x1 = sum(temp_centroid_x1) / len(temp_centroid_x1)
                temp.append(new_centroid_x1)
                
                #club x2 samples with same assigned class labels 
                for k,v in sample_class_dict_2.items():
                    if v == c:
                        temp_centroid_x2.append(k)
                    else:
                        continue
                #print(len(temp_centroid_x2))
                #update x2 part of centroid
                # print("c PRINTING LENtemp_centroid_x2:", c, len(temp_centroid_x2))
                #update x2 part of centroid

                new_centroid_x2 = sum(temp_centroid_x2) / len(temp_centroid_x2)
                temp.append(new_centroid_x2)
                
                #club x3 samples with same assigned class labels 
                for k,v in sample_class_dict_3.items():
                    if v == c:
                        temp_centroid_x3.append(k)
                    else:
                        continue
                #print(len(temp_centroid_x3))
                #update x3 part of centroid
                # print("c PRINTING LENtemp_centroid_x3:", c, len(temp_centroid_x3))
                #update x3 part of centroid

                new_centroid_x3 = sum(temp_centroid_x3) / len(temp_centroid_x3)
                temp.append(new_centroid_x3)
                
                #club x4 samples with same assigned class labels 
                for k,v in sample_class_dict_4.items():
                    if v == c:
                        temp_centroid_x4.append(k)
                    else:
                        continue
                #print(len(temp_centroid_x4))
                #update x4 part of centroid
                # print("c PRINTING LENtemp_centroid_x4:", c, len(temp_centroid_x4))
                #update x4 part of centroid

                new_centroid_x4 = sum(temp_centroid_x4) / len(temp_centroid_x4)
                temp.append(new_centroid_x4)
                
                
                #updated centroid coordinates list
                k_means_centroid_cordinates_new.append(temp) 
                temp = []
                
                #clear temporary list for next iterations
                temp_centroid_x1 = []
                temp_centroid_x2 = []
                temp_centroid_x3 = []
                temp_centroid_x4 = []
                                 
            #print("new cordinates:", k_means_centroid_cordinates_new)    
            
            #compute shift in centroid coordiantes
            error = []
            total = 0
            for index in range(0,len(k_means_centroid_cordinates_new)):
                #print(index)
                c_x1 = k_means_centroid_cordinates_new[index][0] - k_means_centroid_cordinates[index][0]
                c_x2 = k_means_centroid_cordinates_new[index][1] - k_means_centroid_cordinates[index][1]
                c_x3 = k_means_centroid_cordinates_new[index][2] - k_means_centroid_cordinates[index][2]
                c_x4 = k_means_centroid_cordinates_new[index][3] - k_means_centroid_cordinates[index][3]
                c_x1 =pow(c_x1,2)
                c_x2 =pow(c_x2,2)
                c_x3 =pow(c_x3,2)
                c_x4 =pow(c_x4,2)
                total = pow((c_x1 + c_x2 + c_x3 + c_x4),0.5)
                error.append(total)
        
            #increment iteration 
            iter_no = iter_no + 1
            #print("iter_no:", iter_no) 
            
            #compute norm of change in centroid coordinates
            total_t = 0
            for e in error:
                squared = pow(e,2)
                total_t = total_t + squared
                 
                #print("error:", math.sqrt(total_t))
            
            #check if centroid coorinates are shifted 
            if math.sqrt(total_t) <0.001:
                print("iter_no when no change in centroid coordinates:", iter_no)
                #print(assigned_class)  
                print("K means coordinates found, breaking from while loop")  
                break
            else:
                k_means_centroid_cordinates = k_means_centroid_cordinates_new
                continue
                #print("searching for k means coordinates")
            
        
        #print("K means centroid coordinates found")
        #print("K means centroid coordinates at end:", k_means_centroid_cordinates_new)
         
        
        #compute cost function
        diff_x1 = 0
        diff_x2 = 0
        sum_norm = 0
        norm_list = []
        
        
        
        for (x1,v1),(x2,v2),(x3,v3),(x4,v4) in zip(sample_class_dict.items(),sample_class_dict_2.items(), sample_class_dict_3.items(), sample_class_dict_4.items()):
            count = 0
            class_color = None
            for c in k_means_class:#start with every class
                if v1 == c:
                    #find xi - assigned centroid
                    diff_x1 = x1 - k_means_centroid_cordinates_new[count][0]
                    diff_x2 = x2 - k_means_centroid_cordinates_new[count][1]
                    diff_x3 = x3 - k_means_centroid_cordinates_new[count][2]
                    diff_x4 = x4 - k_means_centroid_cordinates_new[count][3]
                    
                else:
                    count = count + 1
                    continue
            #find norm 
            diff_x1 = pow(diff_x1,2)
            diff_x2 = pow(diff_x2,2)
            diff_x3 = pow(diff_x3,2)
            diff_x4 = pow(diff_x4,2)
            sum_norm = diff_x1 + diff_x2 + diff_x3 + diff_x4
            sum_norm = pow(sum_norm,0.5)
            norm_list.append(sum_norm)
        
        J_Cost = 0    
        J_Cost = (1/len(norm_list)) * sum(norm_list)    
        J_list.append(J_Cost)
        
            
        print("Cost for k = " + str(k_means_count) + " :", J_Cost, "\n")
        final_k_with_centroid[k_means_count] = k_means_centroid_cordinates_new
        final_k_with_cost[k_means_count] = J_Cost
        
        # print("Centroid Coordinates for k = " + str(k_means_count) + " :", k_means_centroid_cordinates_new, "\n")
        
        # return k_means_centroid_cordinates_new, k_means_class, sample_class_dict, sample_class_dict_2, sample_class_dict_3, sample_class_dict_4
        """
        
        return k_means_class, sample_class_dict, sample_class_dict_2, sample_class_dict_3, sample_class_dict_4,assigned_class

#print(final_k_with_cost)
#print(J_list)

"""
#get index with elbow k in J_cost list    
j = 0
for j in range(0, len(J_list)):
    diff = J_list[j] - J_list[j+1]
    if (diff >0 and diff <1):
        #print("found index", j)
        break
    else:
        continue
"""

#print outcomes        
# print("OPTIMUM K:", list(final_k_with_cost)[j])
# print("COST at elbow:", final_k_with_cost[list(final_k_with_cost)[j]])
# print("COORDINATES for OPTIMUM K:", "\n", final_k_with_centroid[list(final_k_with_centroid)[j]],"\n")      

k_means = 4

for select_a_kernel in range(0,3):
    #select a kerenl function
    #0: polynomial kernel with d= 2
    #1: polynomial kernel with d= 3
    #2: RBF kernel with sigma = 0.8 
    for rand_init in range (0, 1):
        H_star_unit_row_vector_normalied = Compute_H_star_matrix(select_a_kernel)
        # print("in main filr H_star_unit_row_vector_normalied:", H_star_unit_row_vector_normalied)
        # print("H_star_unit_row_vector_normalied shape MAIN:", H_star_unit_row_vector_normalied.shape)
        # print("H_star_unit_row_vector_normalied TYPE:", type(H_star_unit_row_vector_normalied))
        # print("H_star_unit_row_vector_normalied row1 MAIN:", H_star_unit_row_vector_normalied[0:2,0:4])
        
        # print("Random centroid coordinates initialisation round no. :", str(rand_init+1))
        random.seed(rand_init)
        k_means_class, sample_class_dict, sample_class_dict_2, sample_class_dict_3, sample_class_dict_4, assigned_class = Get_K_Means_Centroid_Cordinates(k_means, H_star_unit_row_vector_normalied)
        # print("Centroid Coordinates for k = " + str(k_means) + " :", k_means_centroid_cordinates_new, "\n")
        
        
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        
        # for (x1,v1),(x2,v2),(x3,v3),(x4,v4) in zip(sample_class_dict.items(),sample_class_dict_2.items(), sample_class_dict_3.items(), sample_class_dict_4.items()):
            # count = 0
        for (x1,v1),(x2,v2) in zip(sample_class_dict.items(),sample_class_dict_2.items()):#, sample_class_dict_3.items(), sample_class_dict_4.items()):
                
            # print("x, v:", x1, v1, x2, v2)# , x3, v3, x4 , v4)
            class_color = None
            for c in k_means_class:#start with every class
                if v1 == c:
                    #find xi - assigned centroid
                    # diff_x1 = x1 - k_means_centroid_cordinates_new[count][0]
                    # diff_x2 = x2 - k_means_centroid_cordinates_new[count][1]
                    if c == '0':
                        class_color = 'green'
                        
                    if c == '1':
                        class_color = 'blue'
                        
                    if c == '2':
                        class_color = 'maroon'
                       
                    if c == '3':
                        class_color = 'orange'
                    
                    #overlay coloured points on same plot 
                    plt.scatter(x1, x2, color = class_color)
                    # img = ax.scatter(x1, x2, x3, c=int(c), cmap=plt.hot())
                    # fig.colorbar(img)
                    
                    plt.xlabel('x1')
                    plt.ylabel('x2')
                    if select_a_kernel == 0:
                        plt.title("Coloured (k=4) clusters for polynomial kernel with d= 2")
                    if select_a_kernel == 1:
                        plt.title("Coloured (k=4) clusters for polynomial kernel with d= 3")
                    if select_a_kernel == 2:
                        plt.title("Coloured (k=4) clusters for RBF kernel with sigma = 0.8")
                    
                else:
                    # count = count + 1
                    continue
        
        #show plot for current set of random centroid centres initialisation
        plt.show()
      
    # Plot_K_Means_Centroids(k, k_means_centroid_cordinates_new)
     
    
print("Spectral clustering algorithm implemented and coloured clusters for kernels are plotted.")        
       

"""
Result:
Plotting coloured clusters (k=4) for fixed rand. init. for ARG MAX based cluster assignment and ARG MAX assignment checking performance
Spectral clustering algorithm implemented and coloured clusters for kernels are plotted.
"""



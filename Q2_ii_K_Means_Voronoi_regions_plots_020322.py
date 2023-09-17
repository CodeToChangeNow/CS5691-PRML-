"""
@author: KALYANI BURANDE
DATE: 02/03/2022
COURSE: CS5691 PRML
ASSIGNMENT 1 : PCA, KERNEL PCA, K-MEANS, SPECTRAL CLUSTERING

PROBLEM STATEMENT:
(2)You are given a data-set with 1000 data points each in R2.
ii. Fix a random initialization. For K = {2; 3; 4; 5}, 
obtain cluster centers according to K-means algorithm using the fixed initialization. 
For each value of K, plot the Voronoi regions associated to each cluster center. 
(You can assume the minimum and maximum value in the data-set to be the range for each component of R2).
"""

print("For k={2; 3; 4; 5} plotting Voronoi regions associated to each cluster center for fixed initialisation.\n")

import numpy as np
import pandas as pd
import random, math 
import matplotlib.pyplot as plt

#Read csv file Using Pandas
data = pd.read_csv("Dataset.csv", header = None)
# df = pd.read_csv(filename)
#print(df)

#Dataset exploration using pd
# print(data.head())
# print(data.iloc[0:5,0])
# print(data.iloc[0:5,1])
# print(data.iloc[:,0])
# print(data.iloc[:,1])

x1 = np.array(data.iloc[:,0])
x2 = np.array(data.iloc[:,1])
X = np.column_stack([x1, x2])

x1_data = x1.tolist()
x2_data = x2.tolist()

# m = number of sample points available in ground truth data
samples_count = len(x1_data) 
# print("number of samples:", samples_count)

#Initialise variables 
k_means_count = 2 # initial no. of k means classes 
final_k_with_centroid = {} #dict to hold k:centroid
final_k_with_cost = {}     #dict to hold k:cost
J_list = []                #list for computed cost values per k 

k_start = 2 #start with these no. of classes
k_stop = 6 #end with these no. of classes

def Get_K_Means_Centroid_Cordinates(k):
    
        #compute cluster centroids for k=4 using k means algorithm 
        random.seed(20)
        
        k_means_count = k
        print("Number of classes = K = " + str(k_means_count))
        k_means_centroid_cordinates = [] #stores all coordinates for all k in a list
        k_means_class = []#store current k class labels in a list
        
        #craete k class labels 
        for k in range(0, k_means_count):
            k_means_class.append(str(k))   
        #print(k_means_class)    
        
        #generate random centroid coordinates
        temp = [] #holds centroid values per k 
        for ele in range (0,k_means_count):
            centroid_x1 = random.uniform(min(x1_data),max(x1_data))
            centroid_x2 = random.uniform(min(x2_data),max(x2_data))
            temp.append(centroid_x1)
            temp.append(centroid_x2)
            k_means_centroid_cordinates.append(temp)
            temp = []
        
        print("k_means_centroid_cordinates at start:", k_means_centroid_cordinates)  
        #print(k_means_centroid_cordinates[0][0])  
        #print(k_means_centroid_cordinates[0][1])  
        #print(k_means_centroid_cordinates[1][0])  
        #print(k_means_centroid_cordinates[1][1])  
        
        #search coordinate for current k 
        iterations = 100
        iter_no = 0
        
        while(iter_no < iterations):
            distance_from_centroid_each_sample = []
            assigned_class = []
            sample_class_dict = {}#create dict, x1:assigned class
            sample_class_dict_2 = {}#create dict, x2:assigned class
            
            #assign nearest class label to each sample
            for sample in range(0, samples_count):
                #for every sample do this
                for centroids in k_means_centroid_cordinates:       
                    #print("centroids:", centroids)
                    #print(centroids[0])
                    #print(centroids[1])
                    #compute distance of label to each centroids
                    distance_1 = pow((x1_data[sample] - centroids[0]),2)
                    distance_2 = pow((x2_data[sample] - centroids[1]),2)
                    dist = pow((distance_1 + distance_2),0.5)
                    distance_from_centroid_each_sample.append(dist)
                #print("distance_from_centroid_each_sample:", distance_from_centroid_each_sample)
                min_distance = min(distance_from_centroid_each_sample)
                min_index = distance_from_centroid_each_sample.index(min_distance)
                #print("min_index:", min_index)
                #print("value at min_index:", k_means_class[min_index], "\n")
                
                #assign class to sample corres. to centroid with smallest distance
                sample_class_dict[x1_data[sample]] =k_means_class[min_index]
                sample_class_dict_2[x2_data[sample]] =k_means_class[min_index]
                
                assigned_class.append(k_means_class[min_index])
                distance_from_centroid_each_sample = []
            
        #    print(sample_class_dict)   
        #    print(sample_class_dict_2) 
        #    print(assigned_class)      
            
            #craete new list to store updated class coordinates
            k_means_centroid_cordinates_new = []
            
            #temporary lists for computing new class coordiantes
            temp_centroid_x1 = []
            temp_centroid_x2 = []    
            temp = []                    
            for c in k_means_class:#start with every class
                #club x1 samples with same assigned class labels 
                for k,v in sample_class_dict.items():
                    if v == c:
                        temp_centroid_x1.append(k)
                    else:
                        continue
                #print(len(temp_centroid_x1))
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
                new_centroid_x2 = sum(temp_centroid_x2) / len(temp_centroid_x2)
                temp.append(new_centroid_x2)
                
                #updated centroid coordinates list
                k_means_centroid_cordinates_new.append(temp) 
                temp = []
                
                #clear temporary list for next iterations
                temp_centroid_x1 = []
                temp_centroid_x2 = []
                                 
            #print("new cordinates:", k_means_centroid_cordinates_new)    
            
            #compute shift in centroid coordiantes
            error = []
            total = 0
            for index in range(0,len(k_means_centroid_cordinates_new)):
                #print(index)
                c_x1 = k_means_centroid_cordinates_new[index][0] - k_means_centroid_cordinates[index][0]
                c_x2 = k_means_centroid_cordinates_new[index][1] - k_means_centroid_cordinates[index][1]
                c_x1 =pow(c_x1,2)
                c_x2 =pow(c_x2,2)
                total = pow((c_x1 + c_x2),0.5)
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
    
        for (x1,v1),(x2,v2) in zip(sample_class_dict.items(),sample_class_dict_2.items()):
            count = 0
            for c in k_means_class:#start with every class
                if v1 == c:
                    #find xi - assigned centroid
                    diff_x1 = x1 - k_means_centroid_cordinates_new[count][0]
                    diff_x2 = x2 - k_means_centroid_cordinates_new[count][1]
                else:
                    count = count + 1
                    continue
            #find norm 
            diff_x1 = pow(diff_x1,2)
            diff_x2 = pow(diff_x2,2)
            sum_norm = diff_x1 + diff_x2
            sum_norm = pow(sum_norm,0.5)
            norm_list.append(sum_norm)
        
        J_Cost = 0    
        J_Cost = (1/len(norm_list)) * sum(norm_list)    
        J_list.append(J_Cost)
            
        # print("Cost for k = " + str(k_means_count) + " :", J_Cost, "\n")
        final_k_with_centroid[k_means_count] = k_means_centroid_cordinates_new
        final_k_with_cost[k_means_count] = J_Cost
        
        # print("Centroid Coordinates for k = " + str(k_means_count) + " :", k_means_centroid_cordinates_new, "\n")
        
        return k_means_centroid_cordinates_new
    
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

from scipy.spatial import Voronoi, voronoi_plot_2d 
for k in range(k_start,k_stop): 
    
    k_means_centroid_cordinates_new = Get_K_Means_Centroid_Cordinates(k)
    print("Centroid Coordinates for k = " + str(k) + " :", k_means_centroid_cordinates_new, "\n")
    nd_array_centroids_cordinates =np.array(k_means_centroid_cordinates_new) #ppoints for vornoi
    # print(nd_array_centroids_cordinates)

    
    if k != 2:
        voronoi = Voronoi(nd_array_centroids_cordinates)
        figure = voronoi_plot_2d(voronoi)
        # fig = voronoi_plot_2d(vor, show_vertices=False, line_colors='orange', line_width=2, line_alpha=0.6, point_size=2)
        # plt.scatter(x1_data, x2_data, color = 'hotpink')
        
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title("K means voronoi and centroids plot for k: " + str(k))
        # plt.legend(loc='upper left')
        plt.show()
    
print("K means algorithm complete and voronoi region per cluster is plotted.")        
       

"""
Result:

For k={2; 3; 4; 5} plotting Voronoi regions associated to each cluster center for fixed initialisation.

Number of classes = K = 2
k_means_centroid_cordinates at start: [[6.86220537154993, 3.5222883647252914], [4.349440424537535, 7.454814750026806]]
iter_no when no change in centroid coordinates: 18
K means coordinates found, breaking from while loop
Centroid Coordinates for k = 2 : [[-0.17612262884333818, -2.0978062570381213], [0.37947054574132494, 4.517563259493672]] 

Number of classes = K = 3
k_means_centroid_cordinates at start: [[6.86220537154993, 3.5222883647252914], [4.349440424537535, 7.454814750026806], [-4.801486384625333, 2.6123143309081804]]
iter_no when no change in centroid coordinates: 23
K means coordinates found, breaking from while loop
Centroid Coordinates for k = 3 : [[3.0721759502074675, -3.8048026333333342], [1.8811809291666672, 5.003905020920502], [-2.2964882842003846, -0.550933613294798]] 

Number of classes = K = 4
k_means_centroid_cordinates at start: [[6.86220537154993, 3.5222883647252914], [4.349440424537535, 7.454814750026806], [-4.801486384625333, 2.6123143309081804], [6.84967171852394, 6.869770332696232]]
iter_no when no change in centroid coordinates: 39
K means coordinates found, breaking from while loop
Centroid Coordinates for k = 4 : [[-2.6642628597560973, -6.046244171779142], [-2.2337890198675496, 6.254868000000001], [-0.7684418494505494, -0.00814185780219783], [4.8864465217391295, 0.2080794260869565]] 

Number of classes = K = 5
k_means_centroid_cordinates at start: [[6.86220537154993, 3.5222883647252914], [4.349440424537535, 7.454814750026806], [-4.801486384625333, 2.6123143309081804], [6.84967171852394, 6.869770332696232], [0.8534949569755703, -5.786236153338187]]
iter_no when no change in centroid coordinates: 21
K means coordinates found, breaking from while loop
Centroid Coordinates for k = 5 : [[4.2501576047904175, -3.378873766467067], [-5.051982719298245, 4.811431149122807], [-0.2019555684782609, 0.1599096754347826], [4.0799194375, 5.3276771653543324], [-4.29908403816794, -5.673527384615382]] 

K means algorithm complete and voronoi region per cluster is plotted.
"""



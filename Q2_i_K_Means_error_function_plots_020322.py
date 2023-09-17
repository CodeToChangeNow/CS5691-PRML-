"""
@author: KALYANI BURANDE
DATE: 02/03/2022
COURSE: CS5691 PRML
ASSIGNMENT 1 : PCA, KERNEL PCA, K-MEANS, SPECTRAL CLUSTERING

PROBLEM STATEMENT:
(2)You are given a data-set with 1000 data points each in R2.
i. Write a piece of code to run the algorithm studied in class for the K-means problem with k = 4 . 
Try 5 different random initialization and plot the error function w.r.t iterations in each case. 
In each case, plot the clusters obtained in different colors.
"""

print("For k=4 plotting the error function w.r.t iterations in each case for 5 different initialisations.\n")

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
k_means_count = 4 # initial no. of k means classes 
final_k_with_centroid = {} #dict to hold k:centroid
final_k_with_cost = {}     #dict to hold k:cost
J_list = []                #list for computed cost values per k 


def Get_K_Means_Centroid_Cordinates(k, rand_init):
        
        #compute cluster centroids for k=4 using k means algorithm 
        k_means_count = k
        print("Number of classes = K = " + str(k_means_count))
        k_means_centroid_cordinates = [] #stores all coordinates for all k in a list
        k_means_class = []#store current k class labels in a list
        
        #craete k class labels 
        for k in range(0, k_means_count):
            k_means_class.append(str(k))   
        print("K-means class labels are:", k_means_class)    
        
        #generate random centroid coordinates
        temp = [] #holds centroid values per k 
        for ele in range (0,k_means_count):
            centroid_x1 = random.uniform(min(x1_data),max(x1_data))
            centroid_x2 = random.uniform(min(x2_data),max(x2_data))
            temp.append(centroid_x1)
            temp.append(centroid_x2)
            k_means_centroid_cordinates.append(temp)
            temp = []
        
        print("\nk_means_centroid_cordinates random intiialisations at start:\n", k_means_centroid_cordinates)  
        #print(k_means_centroid_cordinates[0][0])  
        #print(k_means_centroid_cordinates[0][1])  
        #print(k_means_centroid_cordinates[1][0])  
        #print(k_means_centroid_cordinates[1][1])  
        
        #search coordinate for current k 
        iterations = 100
        iter_no = 0
          
        fig, ax = plt.subplots(figsize=(10, 6))
        
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
        
        # upto this point in code, all the datapoints are assigned with nearest class labels 
            
            #create new list to store updated class coordinates
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
                                 
            # print("new cordinates:", k_means_centroid_cordinates_new)  
            # print("new cordinates LENGTH :", len(k_means_centroid_cordinates_new), iter_no+1)
            
            #compute cost function = sum over all datapoints (squared norm(distance between datapoint and current centroid cordinates computed for this iteration))
            #compute cost function
            diff_x1 = 0
            diff_x2 = 0
            sum_norm = 0
            norm_list = []
            
            
            for (x1,v1),(x2,v2) in zip(sample_class_dict.items(),sample_class_dict_2.items()):
                count = 0
                # class_color = None
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
            
            ax.scatter(iter_no + 1, sum(norm_list), color = 'darkblue')
            plt.xlabel("Iteration no.")
            plt.ylabel("Error Function value")
            plt.title("Error function vs. Iteration count for Random initialisation no:" + str(rand_init+1))
            # print("NORM LIST:", norm_list)
            
            
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
            # class_color = None
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
        # print("J_list:", J_list)
        
            
        print("Cost for k = " + str(k_means_count) + " :", J_Cost, "\n")
        final_k_with_centroid[k_means_count] = k_means_centroid_cordinates_new
        final_k_with_cost[k_means_count] = J_Cost
        
        # print("Centroid Coordinates for k = " + str(k_means_count) + " :", k_means_centroid_cordinates_new, "\n")
        
        return k_means_centroid_cordinates_new, k_means_class, sample_class_dict, sample_class_dict_2
    
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

for rand_init in range (0, 5):
    print("Random centroid coordinates initialisation round no. :", str(rand_init+1))
    random.seed(rand_init)
    k_means_centroid_cordinates_new, k_means_class, sample_class_dict, sample_class_dict_2 = Get_K_Means_Centroid_Cordinates(k_means, rand_init)
    print("Final K-means Centroid Coordinates for k = " + str(k_means) + " :", k_means_centroid_cordinates_new, "\n")

    #show plot for current set of random centroid centres initialisation
    plt.show()
    
       
print("K means algorithm complete and cluster coordinates are found.")        
       
"""
Result:

For k=4 plotting the error function w.r.t iterations in each case for 5 different initialisations.

Random centroid coordinates initialisation round no. : 1
Number of classes = K = 4
K-means class labels are: ['0', '1', '2', '3']

k_means_centroid_cordinates random intiialisations at start:
 [[5.756580849468129, 4.813552433432495], [-1.8983669644045245, -4.173716460623965], [-0.26022289472224713, -1.544060131828], [4.66169441726251, -3.3741804534993802]]
iter_no when no change in centroid coordinates: 12
K means coordinates found, breaking from while loop
Cost for k = 4 : 3.5425700318746594 

Final K-means Centroid Coordinates for k = 4 : [[1.0156661506849314, 6.710788965517245], [-4.345007794326241, -5.472855785714283], [-1.1127009084362143, 0.4966394335390945], [4.427893788546253, -1.9884811982378858]] 

Random centroid coordinates initialisation round no. : 2
Number of classes = K = 4
K-means class labels are: ['0', '1', '2', '3']

k_means_centroid_cordinates random intiialisations at start:
 [[-7.067414569207978, 6.425003655250009], [4.3000515060271365, -4.243010901653608], [-0.5462946095760017, -0.7416255160066587], [2.2739948838594586, 5.367676575269684]]
iter_no when no change in centroid coordinates: 46
K means coordinates found, breaking from while loop
Cost for k = 4 : 3.5253211232369814 

Final K-means Centroid Coordinates for k = 4 : [[-3.5636105255474435, 5.7831857352941185], [-1.0583282748815168, -5.669682857142857], [-0.9039476957040571, 0.08973698019093086], [4.679297854077253, 1.559512738197425]] 

Random centroid coordinates initialisation round no. : 3
Number of classes = K = 4
K-means class labels are: ['0', '1', '2', '3']

k_means_centroid_cordinates random intiialisations at start:
 [[7.772356967455785, 8.233014779949233], [-8.472754023169971, -7.30812326478395], [5.595427488456922, 4.417630727132849], [2.6015659152111077, -3.287308907985585]]
iter_no when no change in centroid coordinates: 16
K means coordinates found, breaking from while loop
Cost for k = 4 : 3.5737537419467182 

Final K-means Centroid Coordinates for k = 4 : [[4.140592404371582, 3.926456617486338], [-4.046317753731342, -5.833121052631576], [-3.5858763268292693, 3.564682843627453], [1.0869941914225938, -1.4081289506276151]] 

Random centroid coordinates initialisation round no. : 4
Number of classes = K = 4
K-means class labels are: ['0', '1', '2', '3']

k_means_centroid_cordinates random intiialisations at start:
 [[-5.196339852406896, 0.9645329641998561], [-2.812524714558415, 2.0395167590865846], [1.8067215523435092, -7.656477668178359], [-9.256279488523194, 6.245548193291567]]
iter_no when no change in centroid coordinates: 13
K means coordinates found, breaking from while loop
Cost for k = 4 : 3.625405095862684 

Final K-means Centroid Coordinates for k = 4 : [[-4.402206505376343, -3.7192917736559137], [1.0538704630390143, 1.1091417117043123], [3.4571089347826076, -4.127362956284155], [-2.311413580419581, 6.3647169014084515]] 

Random centroid coordinates initialisation round no. : 5
Number of classes = K = 4
K-means class labels are: ['0', '1', '2', '3']

k_means_centroid_cordinates random intiialisations at start:
 [[-5.230953475297064, -6.978662256332194], [-2.341090109329796, -6.045673380665233], [-8.292804114478766, -1.604267101935391], [7.084627055685747, 5.578906488558314]]
iter_no when no change in centroid coordinates: 10
K means coordinates found, breaking from while loop
Cost for k = 4 : 3.58169399311765 

Final K-means Centroid Coordinates for k = 4 : [[-4.046317753731342, -5.833121052631576], [1.1020382536534443, -1.4092766981210856], [-3.6225624384236466, 3.5682059118226617], [4.0745154239130414, 3.9527265628415296]] 

K means algorithm complete and cluster coordinates are found.
"""

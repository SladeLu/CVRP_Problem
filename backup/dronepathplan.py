#one dimension
import numpy as np
import math as mh
import matplotlib.pyplot as plt
import itertools

num_drone = 1
num_point = 9 #not include P0

# Using Dji M600 drone parameter
# 6xTB48S battery, 680g each;
# drone weight 10,000g(including battery);
# max take off weight 15,500g(including drones);
# thus capacity is 5,500g
weight_capacity = 1 #not include drone weight
weight_drone =  2 * weight_capacity

Energy_max = 3

# % of the weight of categories  
weight = np.random.rand(num_point)*weight_capacity
# weight = [0.2,0.5,0.7,0.3,0.8,0.1]

# distance from 0-1
position = np.random.rand(num_point,2)
# position.sort()
# position = [0.1,0.2,0.4,0.5,0.8,0.9]

def get_linear_k(Emax,dmin,wmax,wd):
    # Emax>=E(dx,w)+E(dx,wd)
    # Emax>=k*d(w+wd)
    # k<=Emax/d(w+wd)
    return mh.floor((Emax/dmin/(wmax+2*wd))*10000)/10000
linear_k = get_linear_k(1,position[0][-1],weight_capacity,weight_drone)
# linear_k = 0.2

def Distance_fun(p1,p2):
    # return abs(p1-p2)
    # print("P10=",p1[0],"P20=",p2[0],"P11=",p1[1],"P21=",p2[1],"xxx=",mh.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1])))
    return mh.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))

def Energy_fun(d,w):
    return linear_k*d*(w+weight_drone)   
    # return d*mh.tan((w+weight_drone))

def Energy_goback(d):
    return Energy_fun(d,0)

print(linear_k)
print(weight)
print(position)

# plt.xlabel('distance')
# plt.ylabel('weight')             # 设置坐标轴标签
# plt.show()

# def Violent_fun(pos,w,cap):
#     num = len(pos)
#     trip = num
#     pointnum_list = []

#     for i in range(num):
#         pointnum_list.append(i)

#     last_best_route = []
#     last_best_cost = 99999

#     # for i in range(trip):
#     for j in itertools.permutations(pointnum_list,2):
#         j = list(j)
#         weight_sum = 0

#         for x in j:
#             weight_sum+=w[x]

#         if weight_sum>cap:
#             # print("over weight",j,"weight is",weight_sum)
#             continue

#         this_route_cost = Energy_fun(Distance_fun(0,j[0]),weight_sum)

#         count = 0
#         for p in j:
#             weight_sum-=w[count]
#             this_route_cost+=Energy_fun(Distance_fun(pos[count],pos[p]),weight_sum)
#             count+=1

#         this_route_cost+=Energy_fun(Distance_fun(j[-1],0),0)

#         j.insert(0,-1)
#         j.insert(len(j),-1)
#         if this_route_cost>energy_max:
#             print("over energy",j,"energy is",this_route_cost)
#             continue
#         elif this_route_cost<last_best_cost:
#             last_best_cost=this_route_cost
#             last_best_route = j
#             print("update new best",j,"cost is",last_best_cost)
#         else:
#             print("not the best",j,"cost is",this_route_cost)
                
# Violent_fun(position,weight,weight_capacity)
def Greedy_fun(pos,w,C,Q):
    pos_len = len(pos)
    index_list = range(0,pos_len)
    best_distance = 9999
    best_route = []
    for i in itertools.permutations(index_list,pos_len):
        i = list(i)
        # i.insert(0,-1)
        # i.append(-1)
        # print(i)
        distance_sum = 0
        route = []
        while len(i)!=0:
            weight_sum = 0
            temp_list = []
            for j in i:  
                if weight_sum+w[j]<=C:
                    weight_sum += w[j]
                    temp_list.append(j)
                else:
                    break
            # print("templist is",temp_list)
            Energy_cost = 0
            weight_temp = weight_sum
            while True:
                # P0 - P1
                Energy_cost = Energy_fun(Distance_fun([0,0],pos[temp_list[0]]),weight_temp)
                for j in range(0,len(temp_list)-2):
                    weight_temp -= w[j]
                    Energy_cost += Energy_fun(Distance_fun(pos[temp_list[j]],pos[temp_list[j+1]]),weight_temp)
                # PN - P0
                Energy_cost += Energy_fun(Distance_fun(pos[temp_list[len(temp_list)-1]],[0,0]),0)

                if Energy_cost > Q:
                    #print("over energy",j,"energy is",this_route_cost)
                    delt = temp_list.pop()
                    weight_temp = weight_sum
                    weight_temp -= w[len(temp_list)]
                    distance_sum = 0
                else:
                    break
            
            distance_sum += Distance_fun([0,0],pos[temp_list[0]])
            for j in range(0,len(temp_list)-2):
                    distance_sum += Distance_fun(pos[temp_list[j]],pos[temp_list[j+1]])
            distance_sum += Distance_fun(pos[temp_list[len(temp_list)-1]],[0,0])
            
            route.append(temp_list)
            for j in temp_list:
                i.remove(j)

        # print("the route is ",route,"\ndistance is ",distance_sum)
        if distance_sum < best_distance:
            best_distance = distance_sum
            best_route = route
    return best_distance,best_route

def DP_fun(pos,w,C,Q):
    pass

dis,route = Greedy_fun(position,weight,weight_capacity,Energy_max) 
print(dis,route)
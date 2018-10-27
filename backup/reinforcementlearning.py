import numpy as np
import math as mh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

num_drone = 1
num_point = 99 #not include P0

# Using Dji M600 drone parameter
# 6xTB48S battery, 680g each;
# drone weight 10,000g(including battery);
# max take off weight 15,500g(including drones);
# thus capacity is 5,500g

weight_capacity = 1 #not include drone weight
weight_drone =  1.82 * weight_capacity

Energy_max = 3

# % of the weight of categories  
weight = np.random.rand(num_point)*weight_capacity
# weight = [0.2,0.5,0.7,0.3,0.8,0.1]

# distance from 0-1
position = np.random.rand(2,num_point)

# print(len(position[0]),len(position[1]))
fig = plt.figure()
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
ax.scatter(position[0],position[1],weight)  # 绘制数据点
ax.set_zlabel('Weight')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()

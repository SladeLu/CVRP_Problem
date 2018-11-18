import matplotlib.pyplot as plt
import math
import numpy as np

class CVRPEnv_v2():
    def __init__(self,wc=1,wd=1.82,em=50,pointnum = 100):
        # Using Dji M600 drone parameter
        # 6xTB48S battery, 680g each;
        # drone weight 10,000g(including battery);
        # max take off weight 15,500g(including drones);
        # thus capacity is 5,500g

        self.weight_capacity = wc #not include drone weight
        self.weight_drone =  wd * self.weight_capacity
        self.Energy_max = em

        self.pointlist = []
        self.zeropoint = WPoint(0,0,0)

        self.tensor_list = []

        #initial points
        while len(self.pointlist)<pointnum:
            # % of the weight of categories  
            weight = np.around(np.random.rand(1)*self.weight_capacity,2)
            # distance from 0-1
            position = np.around(np.random.rand(2),2)
            # temp = WPoint(position[0],position[1],weight)
            temp = observation_tensor(position[0],position[1],weight,False,0,0)

            self.tensor_list.append(temp)
            # self.pointlist.append(temp)
            if len(self.tensor_list)==pointnum-1:
                self.tensor_list = self._getnextaction(initing=True).copy()
            
    def _getnextaction(self,initing = False):
        if initing:
            pass
        else:
            pass
        for each in self.tensor_list:
            #计算两点间距离
            dis = self.position.distance(each.pos_x,each.pos_y)
            #计算载重
            weight = self.weight_now + each.pos_w
            #计算能量消耗
            energy = self.energy_now + self.Energy_fun(dis,weight+self.weight_drone)

            

    def reset(self):
        self.done = False

        self.pointavailable = self.pointlist.copy()
        self.pointvisited = []

        self.position = self.zeropoint
        self.energy_now = 0
        self.weight_now = 0
        self.distance_now = 0

        self.route_this =[]
        self.route_this.append(self.position)

        self.nextaction = self._getnextaction().copy()
        self.route = []

        self.reward = 0
        self.temp_dis = 0

        self.inf = ''
        print("start from P0(0.00, 0.00)")
        self.inf = "start from P0(0.00, 0.00)"

        self.observation = observation(self.route,self.nextaction,self.distance_now,self.position,\
        self.energy_now/self.Energy_max,self.weight_now/self.weight_capacity)

        return  self.observation


class observation_tensor(object):
    def __init__(self,pos_x,pos_y,pos_w,beenthrough,weight_left,energy_left):
        pass
        
class WPoint():
    def __init__(self,xParam = 0.0,yParam = 0.0,weight = 0):
        self.x = xParam
        self.y = yParam
        self.weight = weight

    def __str__(self):
        return "\n(%.2f, %.2f) weight=%.2f"% (self.x ,self.y,self.weight)
    
    def distance (self,x,y):
        xDiff = self.x - x
        yDiff = self.y - y
        return math.sqrt(xDiff ** 2 + yDiff ** 2)

    def __repr__(self):
        return str(self)

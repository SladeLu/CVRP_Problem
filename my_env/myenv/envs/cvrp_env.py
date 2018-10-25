import gym
from gym import error, spaces, utils
from gym.utils import seeding
import matplotlib.pyplot as plt
import math
import numpy as np
# from myenv.envs.weightpoint import WPoint

class CVRPEnv(gym.Env):
    # metadata = {'render.modes': ['human']}

    def __init__(self):
        self.setstaticbackground()
        self.enableui = False

    def step(self, action):
        self._take_action(action)
        # self.status = self.env.step()
        reward = self._get_reward()
        distance_total = self.distance_now
        ob = self.getState()
        rt = self.route
        if len(ob)==0:
            done = True
            print("UAV DOWN")
        else:
            done = self.done
        # episode_over = self.status != hfo_py.IN_GAME
        return ob, reward, done, distance_total, rt,{}

    def seed(self):
        pass

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

        self.state = self.getState().copy()
        self.route = []

        self.reward = 0

        self.temp_dis = 0


        print("start from P0(0.00, 0.00)")
        if self.enableui:
            self.inf = "start from P0(0.00, 0.00)"
            self.UpdateMap()
        
    def getState(self):
        return self._getState()
    # def render(self, mode='human', close=False):
    #     pass

    def _take_action(self, destination):
        #计算两点间距离
        dis = self.position.distance(destination)
        self.temp_dis = dis
        #计算载重
        self.weight_now += destination.weight
        #计算能量消耗
        self.energy_now += self.Energy_fun(dis,self.weight_now+self.weight_drone)

        self.distance_now += dis
        #将此点加入第j轮
        self.route_this.append(destination)

        #目标点是零点
        if destination is self.zeropoint:
            #充电
            self.energy_now = 0
            #负载清零
            self.weight_now = 0

            if len(self.route_this) > 1:
                #将第j轮加入全局的路径并清空
                self.route.append(self.route_this)
                self.route_this = []
                self.route_this.append(destination)
                self.state = self.pointavailable.copy()

            #没有点可以走了，游戏结束(且回到了原点)
            if len(self.pointavailable) == 0:
                self.done = True

        #现在不是零点
        else:
            #将目标点标志为已访问
            self.route_this.append(destination)
            self.pointvisited.append(destination)
            self.pointavailable.remove(destination)
            self.state = self._getState().copy()

        #将位置移到目标点
        self.position = destination
        print("\ngo to point",destination)
        self.inf = "go to point"+str(destination)
        if self.enableui:
            self.UpdateMap()

    def _getState(self):
        statu = []
        temp_list = self.pointavailable.copy()
        temp_list.append(self.zeropoint)
        for each in temp_list:
            #计算两点间距离
            dis = self.position.distance(each)

            #计算载重
            weight = self.weight_now + each.weight
            #计算能量消耗
            energy = self.energy_now + self.Energy_fun(dis,weight+self.weight_drone)

            if energy <= self.Energy_max and weight <= self.weight_capacity:
                statu.append(each)

        for each in statu:
            dis = each.distance(self.zeropoint)
            #计算载重
            weight = self.weight_now + each.weight
            #计算能量消耗
            energy = self.energy_now + self.Energy_fun(dis,weight+self.weight_drone)

            if energy > self.Energy_max or weight > self.weight_capacity:
                statu.remove(each)
        self.state = statu
        return statu

    def _get_reward(self):
        if self.position is self.zeropoint and self.reward != 0:
            self.reward -= 50
        else:
            self.reward -= 10 * self.temp_dis
        return self.reward

    def setstaticbackground(self,wc=1,wd=1.82,em=50,pointnum = 100):
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

        # % of the weight of categories  
        weight = np.around(np.random.rand(pointnum)*self.weight_capacity,2)

        # distance from 0-1
        position = np.around(np.random.rand(pointnum,2),2)

        #initial points
        for x in range(pointnum):
            temp = WPoint(position[x][0],position[x][1],weight[x])
            self.pointlist.append(temp)
        # print(str(self.pointlist))
    
    def Energy_fun(self,d,w):
        return d*(w**(2)+w**(1.5))

    def UpdateMap(self):
        plt.clf()
        plt.title("DroneMap")
        plt.xlim((0, 1.05))
        plt.ylim((0, 1.05))
        if not self.done:
            plt.text(1, 1,self.inf, style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':5})
        else:
            self.inf = 'Game over,reward is'+str(reward)
            plt.text(1, 1,self.inf, style='italic',bbox={'facecolor':'red', 'alpha':0.5, 'pad':5})
        # Display weight
        list_x,list_y,list_w= [],[],[]
        for p in self.pointlist:
            list_x.append(p.x)
            list_y.append(p.y)
            list_w.append(p.weight)
        for index in range(len(list_w)):
           plt.annotate(list_w[index], xy = (list_x[index], list_y[index]))

        # Blue is visited
        list_x,list_y= [],[]
        for p in self.pointvisited:
            list_x.append(p.x)
            list_y.append(p.y)
        plt.scatter(list_x,list_y, s = 30,alpha = 0.5,c='blue')

        # Red is can be reach
        list_x2,list_y2 = [],[]
        st = self.getState()
        for p in st:
            list_x2.append(p.x)
            list_y2.append(p.y)
        plt.scatter(list_x2,list_y2, s = 20,alpha = 0.5,c='red')

        # Black is unseen
        list_x3,list_y3 = [],[]
        for p in self.pointlist:
            if p not in self.pointvisited and p not in st:
                list_x3.append(p.x)
                list_y3.append(p.y)
        plt.scatter(list_x3,list_y3, s = 20,alpha = 0.5,c='black')

        # draw the arrow
        # for each in self.route:
        #     for i in range(len(each)-1):
        #         plt.arrow(each[i].x,each[i].y,each[i+1].x-each[i].x,each[i+1].y-each[i].y,width=0.002,ec='red')
        each = self.route_this
        if self.zeropoint in self.state and len(self.state)==1:
            each.append(self.zeropoint)
        for i in range(len(each)-1):
            plt.arrow(each[i].x,each[i].y,each[i+1].x-each[i].x,each[i+1].y-each[i].y,width=0.002,ec='red')
            if i == len(each)-3:
                plt.pause(0.001)
        # plt.legend(loc=0)
        plt.draw()
        plt.pause(0.001)
    
    def EnableUI(self):
        self.enableui = True

class WPoint():
    def __init__(self,xParam = 0.0,yParam = 0.0,weight = 0):
        self.x = xParam
        self.y = yParam
        self.weight = weight

    def __str__(self):
        return "\n(%.2f, %.2f) weight=%.2f"% (self.x ,self.y,self.weight)
    
    def distance (self,pt):
        xDiff = self.x - pt.x
        yDiff = self.y - pt.y
        return math.sqrt(xDiff ** 2 + yDiff ** 2)

    def __repr__(self):
        return str(self)
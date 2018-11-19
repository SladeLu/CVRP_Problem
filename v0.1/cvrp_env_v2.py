import matplotlib.pyplot as plt
import math
import numpy as np


class CVRPEnv_v2():
    def __init__(self, wc=1, wd=1.82, em=50, pointnum=5):
        # Using Dji M600 drone parameter
        # 6xTB48S battery, 680g each;
        # drone weight 10,000g(including battery);
        # max take off weight 15,500g(including drones);
        # thus capacity is 5,500g
        self.weight_capacity = wc  # not include drone weight
        self.weight_drone = wd * self.weight_capacity
        self.Energy_max = em

        self.pointlist = []
        self.zeropoint = WPoint(0, 0, 0)
        self.zerotensor = ob_tensor(0, 0, 0, True, 1, 1, False)
        self.tensor_list = [ob_tensor(0, 0, 0, True, 1, 1, True)]
        self.n_action = pointnum

        self.position = self.zeropoint
        self.energy_now = 0
        self.weight_now = 0

        # initial points
        while len(self.tensor_list) < pointnum:
            # % of the weight of categories
            weight = np.around(np.random.rand(1)*self.weight_capacity, 2)
            # distance from 0-1
            position = np.around(np.random.rand(2), 2)
            temp = ob_tensor(position[0], position[1],
                             weight, False, 0, 0, False)

            self.tensor_list.append(temp)
            if len(self.tensor_list) >= pointnum-1:
                self._UpdateTensor(initing=True)

    def _UpdateTensor(self, initing=False):
        for each in self.tensor_list:
            if not each.here:
                # 计算两点间距离
                dis = self.position.distance(each.pos_x, each.pos_y)
                # 计算载重
                weight = self.weight_now + each.pos_w
                # 计算能量消耗
                energy = self.energy_now + \
                    self.Energy_fun(dis, weight+self.weight_drone)
                each.energy_left = (self.Energy_max - energy)/self.Energy_max
                each.weight_left = (self.weight_capacity -
                                    weight)/self.weight_capacity
            
        if initing:
            for each in self.tensor_list:
                if each.weight_left <= 0 and each.energy_left < 0:
                    self.tensor_list.remove(each)
        else:
            pass

    def step(self, action):
        self._take_action(action)
        ob = self.tensor_list
        done = self.done
        reward = self.reward
        return ob,reward,done

    def _take_action(self, action):
        #去过了的点
        if action.beenthrough == 1:
            self._get_reward(0)
            return
        #超过能量/重量的点
        if action.energy_left < 0 or action.weight_left<0:
            self._get_reward(1)
            print("over")
            self.done = True
            return 
        for each in self.tensor_list:
            if each.here:
                each.here = False
                break
        for each in self.tensor_list:
            if each == action:
                each.here = True
                dis = WPoint(each.pos_x, each.pos_y, each.pos_w)
                self.distance_temp = self.position.distance(dis.x,dis.y)
                self.position = dis
                self.route_this.append(each)

                if each.pos_x == 0 and each.pos_y == 0:
                    each.energy_left = 1
                    each.weight_left = 1
                    each.beenthrough = True
                    self.route.append(self.route_this.copy())
                    self.route_this = [ob_tensor(0, 0, 0, True, 1, 1, True)]
                else:
                    self.weight_now = self.weight_capacity - each.weight_left
                    self.energy_now = self.Energy_max - each.energy_left
                    each.beenthrough = True
                self._UpdateTensor()
            elif each.pos_x == 0 and each.pos_y == 0:
                each.beenthrough = False
        self.distance_now += self.distance_temp
        self._get_reward(2)

        for each in self.tensor_list:
            if each.beenthrough is False and (each.pos_x != 0 and each.pos_y != 0):
                self.done = False
                return
        self.done = True

    def _get_reward(self,sta):
        if sta==0:
            pass
        elif sta == 1:
            self.reward -= 10000
        elif sta ==2 :
            self.reward -= self.distance_temp

        pass

    def Energy_fun(self, d, w):
        return d*(w**(2)+w**(1.5))

    def reset(self):
        self.done = False
        self.position = self.zeropoint
        self.energy_now = 0
        self.weight_now = 0
        self.distance_now = 0

        self.route_this = [ob_tensor(0, 0, 0, True, 1, 1, True)]
        self.route = []
        self.reward = 0

        return self.tensor_list


class ob_tensor(object):
    def __init__(self, pos_x, pos_y, pos_w, beenthrough, weight_left, energy_left, here):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_w = pos_w
        self.beenthrough = beenthrough
        self.weight_left = weight_left
        self.energy_left = energy_left
        self.here = here

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        if self.pos_x == other.pos_x and\
                self.pos_y == other.pos_y and\
                self.pos_w == other.pos_w and\
                self.beenthrough == other.beenthrough and\
                self.weight_left == other.weight_left and\
                self.energy_left == other.energy_left and\
                self.here == other.here:
            return True
        return False

    def __str__(self):
        return "\n(%.2f, %.2f),weight=%.2f,beenthrough = %d,weight-left = %0.2f,energy-left = %0.2f drone is here(%d)"\
            % (self.pos_x, self.pos_y, self.pos_w, self.beenthrough, self.weight_left, self.energy_left, self.here)


class WPoint():
    def __init__(self, xParam=0.0, yParam=0.0, weight=0):
        self.x = xParam
        self.y = yParam
        self.weight = weight

    def __str__(self):
        return "\n(%.2f, %.2f) weight=%.2f" % (self.x, self.y, self.weight)

    def distance(self, x, y):
        xDiff = self.x - x
        yDiff = self.y - y
        return math.sqrt(xDiff ** 2 + yDiff ** 2)

    def __repr__(self):
        return str(self)

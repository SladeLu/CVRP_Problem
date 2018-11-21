#!/usr/bin/env python
# coding: utf-8

# In[2]:


import math
import random
import numpy as np
import copy


# In[3]:


class WPoint():
    def __init__(self,xParam,yParam,weight):
        self.x = xParam
        self.y = yParam
        self.weight = weight

    def __str__(self):
        return "\n(%.2f, %.2f) weight=%.2f"% (self.x ,self.y, self.weight)
    
    def info(self):
        info = (self.x, self.y, self.weight)
        return np.array(info)

    def __getitem__(self,index):
        return self.index

    def __setitem__(self,index,value):
        self.index = value
    
    def distance (self,pt):
        xDiff = self.x - pt.x
        yDiff = self.y - pt.y
        return math.sqrt(xDiff ** 2 + yDiff ** 2)

    def __repr__(self):
        return str(self)

class WState(WPoint):
    def __init__(self, xParam, yParam, weight, is_current=False):
        super(WState,self).__init__(xParam = xParam, yParam = yParam, weight = weight)
        self.is_current = is_current

    def state(self):
        state = (self.x, self.y, self.weight, self.is_current)
        return np.array(state)
            
    def __str__(self):
        return "\n(%.2f, %.2f) weight=%.2f current:%d"% (self.x ,self.y, self.weight, self.is_current)

    def __repr__(self):
        return str(self)


# In[19]:


class TestEnv_v2():
    """
    Description:
        This is a multi agent cargo delivery environment    

    observation: 
        Box(point_num,7)
        Num Observation                 Min         Max
        index                           0           point_num
        position-x                      0           x_border
        position-y                      0           y_border
        node weight                     0           max_weight - origin_weight
        is_used                         0           1                        del 
        is_current
        next_weight
        next_power

    Actions:
        Type: Discrete(point_num)
    Reward:
        0---->+inf
        condition           reward
        point not used      1/mileage of this step          
        point is used       0   

    Other value environment keep and calculated
        power remain
        carrying weight 
        time used   

    Starting State:
        start from base with no load and full power

    Episode Termination:
        finish all nodes
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.debug = 0
        self.origin_weight = 8.0
        self.full_power = 100.0
        self.point_num = 100            #include base
        self.site_num = self.point_num - 1
        self.max_weight = 20.0
        self.min_cargo_weight = 0.3
        self.max_cargo_weight = 2.0

        self.pointlist = []
        self.point_state = []
        self._creat_map()
        self.state = np.empty((self.point_num,6),dtype=np.float)
        
        self.reward = 0
        self.action_num = self.point_num
        self.observation_size = np.shape(self.state)

    def step(self, action):
        self.steps += 1
        distance = self.pointlist[action].distance(self.pointlist[self.site])
        power_prediction = self.power - self._power_cost(distance, self.weight)
        weight_prediction = self.weight + self.pointlist[action].weight

        if self.point_state[action].weight == 0 or (action == 0 and self.site == 0):
            self._stay()
        elif power_prediction < 0 or weight_prediction > self.max_weight:
            self._reset_round()
            if self.debug:
                if power_prediction < 0:
                    self.power_reset += 1
                    print("power reset: %d"% (self.power_reset))
                else:
                    self.weight_reset += 1
                    print("weight reset: %d"% (self.weight_reset))
        else:
            self._go_next_point(action)

        return self.state, self.reward, self.done, [self.finished, self.rounds, self.total_distance, self.steps]

    def reset(self):
        self.done = False
        self.site = 0
        self.steps = 0
        self.finished = 0
        self.round_finished = 0
        self.rounds = 0
        self.total_distance = 0
        self.round_distance = 0
        # self.round_reward = 0
        self.round_resets = 0
        self.power_reset = 0
        self.weight_reset = 0
        self.power = self.full_power
        self._reset_site()
        self.round_state = copy.deepcopy(self.point_state)
        self._go_base()
        self._get_state()
        
        return self.state

    def render(self, mode='human', close=False):
        pass

    def _reset_round(self):
        self.point_state = copy.deepcopy(self.round_state)
        self.reward = -10
        # self.reward = -1 * self.round_reward
        # self.round_reward = 0
        self.finished -= self.round_finished
        self.round_finished = 0
        self.total_distance -= self.round_distance
        self.round_distance = 0
        self.round_resets += 1
        self._go_base()
        self._get_state()

    def _stay(self):
        self.reward = -6

    def _go_next_point(self, point_index):
        self.point_state[self.site].is_current = False
        self.point_state[point_index].is_current = True
        self.load += self.point_state[point_index].weight
        distance = self.pointlist[point_index].distance(self.pointlist[self.site])
        self.reward = -1 * distance
        self.total_distance += distance
        self.site = point_index

        if point_index == 0:
            # self.point_state[point_index].weight += self.load
            self._go_base()
            self.round_state = copy.deepcopy(self.point_state)
            # self.round_reward = 0
            if self.debug:
                print("self.round_finished: %d"% (self.round_finished))
            self.round_finished = 0
            self.round_distance = 0
            self.rounds += 1

            if self.finished == self.point_num - 1:
                self.done = True
        else:
            self.finished += 1
            self.round_finished += 1
            self.point_state[point_index].weight = 0
            self.power -= self._power_cost(distance,self.weight)
            # self.round_reward += self.reward
            self.round_distance += distance
            self._update_weight()

        self._get_state()

    def _power_cost(self,d,w):
        return 0.08*d*(w**(2)+w**(1.5))

    def _go_base(self):
        self.site = 0
        self.load = 0
        self._update_weight()
        self.power = self.full_power

    def _update_weight(self):
        self.weight = self.origin_weight + self.load

    def _get_state(self):
        for i in range(self.point_num):
            for j in range(4):
                self.state[i][j] = self.point_state[i].state()[j]

            distance = self.pointlist[i].distance(self.pointlist[self.site])
            self.state[i][4] = self.weight + self.pointlist[i].weight
            self.state[i][5] = self.power - self._power_cost(distance,self.weight)
            # print(distance,self.state[i][4],self.state[i][5])


    def _reset_site(self):
        for i in range(self.point_num):
            self.point_state[i].xParam = self.pointlist[i].info()[0]
            self.point_state[i].yParam = self.pointlist[i].info()[1]
            self.point_state[i].weight = self.pointlist[i].info()[2]
            self.point_state[i].is_current = False
        self.point_state[0].is_current = True

    def _creat_map(self):

        zeropoint = WPoint(0,0,0.1)
        zerostate = WState(0,0,0.1,True)    # give base little weight to make it will back

        # % of the weight of categories  
        weight = np.around(np.random.uniform(self.min_cargo_weight,self.max_cargo_weight,self.site_num), 2)

        # distance from 0-1
        position = np.around(np.random.rand(self.site_num,2), 2)

        #initial points
        self.pointlist.append(zeropoint)
        self.point_state.append(zerostate)
        for x in range(self.site_num):
            temp = WPoint(position[x][0],position[x][1],weight[x])
            self.pointlist.append(temp)
            temp = WState(position[x][0],position[x][1],weight[x])
            self.point_state.append(temp)


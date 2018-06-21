import numpy as np

import utilize

# todo add static variable
class ActMethod:
    def __init__(self):
        self.UF = 'uf'
        self.LB = 'lb'
        self.SP = 'sp'
        self.SRND = 'srnd'
        self.RND = 'rnd'
        self.DRL = 'drl'
        self.NUM = 'num'
        self.FIX = 'fix'

class Explorer:

    def __init__(self, epsilon_begin, epsilon_end, epsilon_steps, dim_act, num_path, seed):
        np.random.seed(seed)
        self.__ep_b = epsilon_begin
        self.__ep_e = epsilon_end
        self.__ep = epsilon_begin
        self.__steps = epsilon_steps
        self.__num_paths = num_path
        self.__dim_act = dim_act
        self.__num_act = utilize.convert_action(utilize.get_base_solution(self.__dim_act), num_path)
        self.__fix_act = utilize.convert_action(utilize.get_fix_solution(), num_path)
        self.__avg_act = utilize.convert_action(np.ones(self.__dim_act), num_path)
        self.__ospf_act = utilize.convert_action(np.zeros(self.__dim_act), num_path)

        self.__ep_last = -1
        self.__act_last = self.__ospf_act

        self.__st = 0

    def cut_convert_act(self, act):
        act = np.clip(act, 0.0001, 2.)
        act = utilize.convert_action(act, self.__num_paths)
        return act

    def get_act_ep(self, action):
        self.__ep -= (self.__ep_b - self.__ep_e) / self.__steps
        # Learn from extrema solution
        tmp = (2. * np.random.random(self.__dim_act) - 1.)
        # tmp *= utilize.get_ext_solution(self.__dim_act, self.__num_paths)

        # Learn from teacher
        if np.random.random() < self.__ep:
            act = self.__num_act + tmp
        else:
            act = action + self.__ep * tmp

        # Learn by myself
        # act = action + self.__ep * (2. * np.random.random(self.__dim_act) - 1.)

        return self.cut_convert_act(act)

    def get_act_uniform(self, action, episode):
        act = action + (1. / (1. + episode)) #*  (2. * np.random.random(self.__dim_act) - 1.)
        return self.cut_convert_act(act)

    def get_act(self, action, episode, flag=None):
        assert flag, 'Please Give Action Flag'

        # self.__st += 1
        # if self.__st <5:
        #     return utilize.get_rnd_solution(self.__dim_act, self.__num_paths)

        if flag == 'uf':
            return self.get_act_uniform(action, episode)
        elif flag == 'drl':
            return self.get_act_ep(action)
        elif flag ==  'rnd':
            return utilize.get_rnd_solution(self.__dim_act, self.__num_paths)
        elif flag == 'srnd':
            if self.__ep_last != episode:
                self.__act_last = utilize.get_rnd_solution(self.__dim_act, self.__num_paths)
                self.__ep_last = episode
            return self.__act_last
        elif flag == 'num':return self.__num_act
        elif flag == 'lb':return self.__avg_act
        elif flag == 'sp':return self.__ospf_act
        elif flag == 'fix':return self.__fix_act
        assert False, 'Action Flag Error'
from socket import*
import datetime
import os
import sys

import utilize
if not hasattr(sys, 'argv'):
    sys.argv  = ['']
import tensorflow as tf
import numpy as np

from Network.actor import ActorNetwork
from Network.critic import CriticNetwork
from ReplayBuffer.replaybuffer import PrioritizedReplayBuffer, ReplayBuffer
from Summary.summary import Summary
from Explorer.explorer import Explorer
from SimEnv.Env import Env
from flag import FLAGS

TIME_STAMP = str(datetime.datetime.now())

SERVER_PORT = getattr(FLAGS, 'server_port')
SERVER_IP = getattr(FLAGS, 'server_ip')

#MULTI_AGENT = getattr(FLAGS, 'multi_agent')

SIM_FLAG = getattr(FLAGS, 'sim_flag')
ACT_FLAG = getattr(FLAGS, 'act_flag')
SEED = getattr(FLAGS, 'random_seed')

DIM_STATE = getattr(FLAGS, 'dim_state')
DIM_ACTION = getattr(FLAGS, 'dim_action')
NUM_PATHS = [3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]

ACTOR_LEARNING_RATE = getattr(FLAGS, 'learning_rate_actor')
CRITIC_LEARNING_RATE = getattr(FLAGS, 'learning_rate_critic')

GAMMA = getattr(FLAGS, 'gamma')
TAU = getattr(FLAGS, 'tau')
ALPHA = getattr(FLAGS, 'alpha')
BETA = getattr(FLAGS, 'beta')
MU = getattr(FLAGS, 'mu')
DELTA = getattr(FLAGS, 'delta')

EP_BEGIN = getattr(FLAGS, 'epsilon_begin')
EP_END = getattr(FLAGS, 'epsilon_end')
EP_ST = getattr(FLAGS, 'epsilon_steps')

ACTION_BOUND = getattr(FLAGS, 'action_bound')

BUFFER_SIZE = getattr(FLAGS, 'size_buffer')
MINI_BATCH = getattr(FLAGS, 'mini_batch')

MAX_EPISODES = getattr(FLAGS, 'episodes')
MAX_EP_STEPS = getattr(FLAGS, 'epochs')

if getattr(FLAGS, 'stamp_type') == '__TIME_STAMP':
    REAL_STAMP = TIME_STAMP
else:
    REAL_STAMP = getattr(FLAGS, 'stamp_type')
DIR_SUM = getattr(FLAGS, 'dir_sum').format(REAL_STAMP)
DIR_RAW = getattr(FLAGS, 'dir_raw').format(REAL_STAMP)
DIR_MOD = getattr(FLAGS, 'dir_mod').format(REAL_STAMP)
DIR_LOG = getattr(FLAGS, 'dir_log').format(REAL_STAMP)

                #reward = np.sum(np.log(np.array(thr_sum)) - DELTA * np.log(np.array(ecnpkt) + 1e-10))

FEAT_SELE = getattr(FLAGS, 'feature_select') # a string shows the features selected
RWD_SELE = getattr(FLAGS, "reward_type") # only for multiagent
AGENT_TYPE = getattr(FLAGS, "agent_type")

DETA_W = getattr(FLAGS, "deta_w")
DETA_L = getattr(FLAGS, "deta_l") # for multiagent deta_w < deta_l
EXP_DEC = getattr(FLAGS, "explo_dec")
BATCH_MON = getattr(FLAGS, "batch_mon")

class DrlAgent:
    def __init__(self, state_init, action_init, dim_state, dim_action, num_paths):
        sess = tf.Session()

        self.__dim_state = dim_state
                #reward = np.sum(np.log(np.array(thr_sum)) - DELTA * np.log(np.array(ecnpkt) + 1e-10))
        self.__dim_action = dim_action

        self.__actor = ActorNetwork(sess, dim_state, dim_action, ACTION_BOUND,
                                    ACTOR_LEARNING_RATE, TAU, num_paths)
        self.__critic = CriticNetwork(sess, dim_state, dim_action,
                                      CRITIC_LEARNING_RATE, TAU, self.__actor.num_trainable_vars)

        self.__prioritized_replay = PrioritizedReplayBuffer(BUFFER_SIZE, MINI_BATCH, ALPHA, MU, SEED)
        self.__replay = ReplayBuffer(BUFFER_SIZE, SEED) # being depretated ?

        self.__summary = Summary(sess, DIR_SUM)
        self.__summary.add_variable(name='throughput')
        self.__summary.add_variable(name='delay')
        self.__summary.add_variable(name='reward')
        self.__summary.add_variable(name='ep-reward')
        self.__summary.add_variable(name='ep-max-q')
        self.__summary.build()
        #self.__summary.write_vars(FLAGS)

        self.__explorer = Explorer(EP_BEGIN, EP_END, EP_ST, dim_action, num_paths, SEED)
        #self.__explorer = Explorer(EP_BEGIN, EP_END, EXP_DEC, dim_action, num_paths, SEED)

        sess.run(tf.global_variables_initializer())
        self.__actor.update_target_paras()
        self.__critic.update_target_paras()

        self.__state_curt = state_init
        self.__action_curt = action_init
        self.__base_sol = utilize.get_base_solution(dim_action) # being depretated ?

        self.__episode = 0
        self.__step = 0
        self.__ep_reward = 0.
        self.__ep_avg_max_q = 0.

        self.__beta = BETA

        self.__detaw = DETA_W
        self.__detal = DETA_L

        # for batch repeated used modified on 7.23
        self.__maxutil_curt = -1. 
        self.__batch_curt = None
        self.__weights_curt = None
        self.__indices_curt = None
        self.__batch_mon = BATCH_MON # batch momentum

    @property
    def timer(self):
        return '| %s '%ACT_FLAG \
               + '| tm: %s '%datetime.datetime.now() \
               + '| ep: %.4d '%self.__episode \
               + '| st: %.4d '%self.__step

    def predict(self, state_new, reward, thr, dly, maxutil):
        self.__step += 1
        self.__ep_reward += reward

        # state_new = np.concatenate((utilize.convert_action(
        #                                 self.__base_sol, NUM_PATHS),
        #                             state_new,)).flatten()
        # print('state', state_new)

        self.__summary.run(feed_dict={
            'throughput': thr,
            'delay': dly,
            'reward': reward,
            'ep-reward': self.__ep_reward,
            'ep-max-q': self.__ep_avg_max_q/MAX_EP_STEPS
        }, step=self.__episode*MAX_EP_STEPS+self.__step)
        print(self.timer)
        print('| Reward: %.4f' % reward,
              '| action: ' + '%.2f ' * self.__dim_action % tuple(self.__action_curt))
        if self.__step >= MAX_EP_STEPS:
            self.__step = 0
            self.__episode += 1
            self.__ep_reward = 0.
            self.__ep_avg_max_q = 0.

        action_original = self.__actor.predict([state_new])[0]

        #print('act_o', action_original)

        action = self.__explorer.get_act(action_original, self.__episode, flag=ACT_FLAG)

        #print('act_s', action)

        # Priority
        target_q = self.__critic.predict_target(
            [state_new], self.__actor.predict_target([state_new]))[0]
        value_q = self.__critic.predict([self.__state_curt], [self.__action_curt])[0]
        grads = self.__critic.calculate_gradients([self.__state_curt], [self.__action_curt])
        td_error = abs(reward + GAMMA * target_q - value_q)

        transition = (self.__state_curt, self.__action_curt, reward, state_new)
        self.__prioritized_replay.add(transition, td_error, abs(np.mean(grads[0]))) # td error.shape = [td_error]
        self.__replay.add(transition[0], transition[1], transition[2], transition[3])

        self.__state_curt = state_new
        self.__action_curt = action

        if len(self.__prioritized_replay) > MINI_BATCH:
            curr_q = self.__critic.predict_target([state_new], self.__actor.predict([state_new]))[0]
            if curr_q[0] > target_q[0]:
                self.train(True, maxutil)
            else:
                self.train(False, maxutil)
            self.__maxutil_curt = maxutil # for batch momentum

        return action

    def train(self, curr_stat, maxutil):
        self.__beta += (1-self.__beta) / EP_ST
        if self.__batch_mon and maxutil < self.__maxutil_curt + 1e-2 and maxutil < 1.0:
            batch = self.__batch_curt
            weights = self.__weights_curt
            indices = self.__indices_curt
        else:
            batch, weights, indices = self.__prioritized_replay.select(self.__beta)
            self.__batch_curt = batch
            self.__weights_curt = weights
            self.__indices_curt = indices
        weights = np.expand_dims(weights, axis=1)

        batch_state = []
        batch_action = []
        batch_reward = []
        batch_state_next = []
        for val in batch:
            try:
                batch_state.append(val[0])
                batch_action.append(val[1])
                batch_reward.append(val[2])
                batch_state_next.append(val[3])
            except TypeError:
                print('*'*20)
                print('--val--', val)
                print('*'*20)
                continue
        #
        # batch_state, batch_action, \
        # batch_reward, batch_state_next \
        #net = tflearn.fully_connected(net, 64, activation='LeakyReLU')
        #     = self.__replay.sample_batch(MINI_BATCH)
        # weights = np.reshape(np.ones(MINI_BATCH), (MINI_BATCH, 1))


        target_q = self.__critic.predict_target(
            batch_state_next, self.__actor.predict_target(batch_state_next))
        value_q = self.__critic.predict(batch_state, batch_action)

        batch_y = []
        batch_error = []
        for k in range(len(batch_reward)):
            target_y = batch_reward[k] + GAMMA * target_q[k]
            batch_error.append(abs(target_y - value_q[k]))
            batch_y.append(target_y)
        #print("batch info:", batch_state, batch_action, batch_y)
        predicted_q, _ = self.__critic.train(batch_state, batch_action, batch_y, weights)

        self.__ep_avg_max_q += np.amax(predicted_q)

        a_outs = self.__actor.predict(batch_state)
        grads = self.__critic.calculate_gradients(batch_state, a_outs)

        print('*'*20)
        print("training_grad:", grads[0])
        print("weights:", weights)
        #print("predicted_q:", predicted_q)
        #print("a_outs:", a_outs)

        # Prioritized
        self.__prioritized_replay.priority_update(indices,
                                                  np.array(batch_error).flatten(),
                                                  abs(np.mean(grads[0], axis=1)))
        
        print("grads[0].shape:", grads[0].shape)
        weighted_grads = weights * grads[0]
        # curr_stat = True means the agent is more probably in a right direction, otherwise in a wrong direction (need a larger gradient to chase other agent)
        if curr_stat:
            weighted_grads *= self.__detaw
        else:
            weighted_grads *= self.__detal
        self.__actor.train(batch_state, weighted_grads)
        #self.__actor.train(batch_state, grads[0])

        self.__actor.update_target_paras()
        self.__critic.update_target_paras()


if not SIM_FLAG:
    # receive the preset information
    
    ns3Server = (SERVER_IP, SERVER_PORT)
    tcpSocket = socket(AF_INET, SOCK_STREAM)
    tcpSocket.connect(ns3Server)
    
    
    msgTotalLen = 0
    msgRecvLen = 0
    msg = ""
    blockSize = 1024;
    BUFSIZE = 1025
    sessionSrc = []
    agents = []
    srcEdgeNumDic = {}
    EDGE_NUM = None
    sessPathUtilNum = [] # utils num, i.e. edge num for each path of each session, shape: [[...], [...], ...]
    sessUtilNum = [] # utils num for each session, shape: [...]

    while True:
        datarecv = tcpSocket.recv(BUFSIZE).decode()
        if len(datarecv) > 0:
            if msgTotalLen == 0:
                totalLenStr = (datarecv.split(';'))[0]
                msgTotalLen = int(totalLenStr) + len(totalLenStr) + 1#1 is the length of ';'
            msgRecvLen += len(datarecv)
            msg += datarecv
            if msgRecvLen < msgTotalLen: 
                continue
            #print(msg)
            msgList = (msg.split(';')[1]).split(',');
            
            #DIM_STATE = len(msgList) # sessionNum*2 only for concentrate type
            EDGE_NUM = int(msg.split(';')[2])
            print("EDGE_NUM %d" % EDGE_NUM)
            DIM_STATE = EDGE_NUM # modified for util model 2018.7.19, now being deperated
            
            DIM_ACTION = 0
            NUM_PATHS = []
            for i in range(len(msgList)):
                if i%2 == 0:
                    sessionSrc.append(int(msgList[i]))
                else:
                    DIM_ACTION += int(msgList[i])
                    NUM_PATHS.append(int(msgList[i]))
            
            # information of preset linked edge for each src node
            srcEdgeNumList = (msg.split(';')[3]).split(',');
            for i in range(len(srcEdgeNumList)):
                pair_ = srcEdgeNumList[i].split(' ')
                srcNode = int(pair_[0])
                srcEdges = int(pair_[1])
                srcEdgeNumDic[srcNode] = srcEdges
            
            # information of preset sess (path) utils
            sessPathUtilNumList = (msg.split(';')[4]).split(',')
            sessUtilNumList = (msg.split(';')[5]).split(',')
            print("----->>>>>>>>>>>>>")
            for i in range(len(sessPathUtilNumList)):
                sessPathUtilNum.append([])
                sessUtilNum.append(int(sessUtilNumList[i]))
                spuTmp = sessPathUtilNumList[i].split(" ")
                for j in range(len(spuTmp)):
                    sessPathUtilNum[i].append(int(spuTmp[j]))
            print("sessPathUtilNum:", sessPathUtilNum)
            print("sessUtilNum:", sessUtilNum)
            
            break

    print("DIM_STATE:", DIM_STATE)
    print("DIM_ACTION:", DIM_ACTION)
    print("NUM_PATHS:", NUM_PATHS)
    print("sessionSrc:", sessionSrc)
    print("srcEdgeNumDic:", srcEdgeNumDic)

    if AGENT_TYPE == "multi_agent":
        AGENT_NUM = max(sessionSrc) + 1
        srcSessNum = [0] * AGENT_NUM
        srcPathNum = [0] * AGENT_NUM
        srcPathUtilNum = [0] * AGENT_NUM # utils num for each src, modified at 7.20
        srcUtilNum = [0] * AGENT_NUM # sum util num for each src (sum util for each path), modified at 7.20
        srcPaths = []
        for i in range(AGENT_NUM):
            srcPaths.append([])
        for i in range(len(sessionSrc)):
            srcSessNum[sessionSrc[i]] += 1
            srcPathNum[sessionSrc[i]] += NUM_PATHS[i]
            srcPaths[sessionSrc[i]].append(NUM_PATHS[i])

            srcUtilNum[sessionSrc[i]] += sessUtilNum[i]
            srcPathUtilNum[sessionSrc[i]] += sum(sessPathUtilNum[i])

        print("srcSessNum", srcSessNum)
        print("srcPathNum", srcPathNum)
        print("srcPaths", srcPaths)
        
        for i in range(AGENT_NUM):
            if (srcSessNum[i] > 0):
                # calculate the state dimmension for each agent
                temp_dim_s = 0
                if(FEAT_SELE[0] == "1"):
                    temp_dim_s += srcPathNum[i]
                if(FEAT_SELE[1] == "1"):
                    temp_dim_s += srcSessNum[i]
                if(FEAT_SELE[2] == "1"):
                    temp_dim_s += srcPathNum[i]
                if(FEAT_SELE[3] == "1"):
                    temp_dim_s += srcSessNum[i]
                if(FEAT_SELE[4] == "1"):
                    if i in srcEdgeNumDic:
                        temp_dim_s += srcEdgeNumDic[i] # loss doesnt being used
                if(FEAT_SELE[5] == "1"):
                    temp_dim_s += srcSessNum[i]
                if(FEAT_SELE[6] == "1"):
                    #temp_dim_s += EDGE_NUM # being deprated now
                    #temp_dim_s += srcPathUtilNum[i]
                    temp_dim_s += srcPathNum[i] * 2 # for max and sum of path util
                if(FEAT_SELE[7] == "1"):
                    temp_dim_s += srcUtilNum[i]
                    #temp_dim_s += srcSessNum[i]

                state = np.zeros(temp_dim_s) 
                action = utilize.convert_action(np.ones(srcPathNum[i]), srcPaths[i])
                agent = DrlAgent(state, action, temp_dim_s, srcPathNum[i], srcPaths[i]) 
            else:
                agent = None
            agents.append(agent)
        action = utilize.convert_action(np.ones(DIM_ACTION), NUM_PATHS)
        ret_c = tuple(action)
    elif AGENT_TYPE == "drl_te":
        state = np.zeros(DIM_STATE)
        action = utilize.convert_action(np.ones(DIM_ACTION), NUM_PATHS)
        print("DIM_STATE:", DIM_STATE, "DIM_ACTION:", DIM_ACTION)
        agent = DrlAgent(state, action, DIM_STATE, DIM_ACTION, NUM_PATHS)
        agents.append(agent)
        ret_c = tuple(action)
    elif AGENT_TYPE == "MCF":
        action = []
        ansfile = open(getattr(FLAGS, "mcf_path"), "r")
        for i in ansfile.readlines():
            action.append(float(i.strip()))
        #action_mcf = action
    #print("ECNpktsList:", ECNpktsList)
    else:
        # for OSPF
        #infact only one path for each session
        action = utilize.convert_action(np.ones(DIM_ACTION), NUM_PATHS)

    if not os.path.exists(DIR_LOG):
        os.mkdir(DIR_LOG)

    file_sta_out = open(DIR_LOG + '/sta.log', 'w', 1)
    file_rwd_out = open(DIR_LOG + '/rwd.log', 'w', 1)
    file_act_out = open(DIR_LOG + '/act.log', 'w', 1)
    file_thr_out = open(DIR_LOG + '/thr.log', 'w', 1)
    file_del_out = open(DIR_LOG + '/del.log', 'w', 1)
    file_util_out = open(DIR_LOG + '/util.log', 'w', 1) # file record the max utility

'''
some notes:
sessionSrc = [src_of_sess1, src_of_sess2, ...]#this is a list of source node of each session; its length is the number of sessions
pacNos = [[],[],[],...]#a list of packet number for each session and each path; the list has two dimensions: session and path
dels = [[],[],[],...]#similar to pacNos, except that the content is time delays for each session and each path
thrs = [[],[],[],...]#tong shang
ECNpkts = [[],[],[],...]#tong shang
SrcEdgeNumDic

When do distributed drl learning, multiple agent=DrlAgent(..) are needed, so the upper codes need to be modified
the state of distributed drl learning also contains two kinds of parameters: delay and throughput #we may add, delete, replace parameters in the future
'''

'''
extra notes:
srcSessNum = [sessnum_of_src1, sessnum_of_src2, ...] the sess_num will be zero if there isnt any session with src as source node
srcPathNum = [pathnum of src1, pathnum_of_src2, ...]
srcPaths = [[pahtnum of sess1_1, pathnum_of_sess1_2],[],....] sessi_j means the jth sess belongs to srci
each agent (may be None) matches a src node
srcEdgeUL: a dict which each src nodes has a list [[util, loss], ...] indicates the uitls and loss for each linked edge originated from the src node
For multi-agent each srcnode will have its own pacNos, dels, thrs, ECNpkts
'''

def split_arg(para):
    paraList = para.split(';')
    pacNosList = paraList[1].split(',')
    delsList = paraList[2].split(',')
    thrsList = paraList[3].split(',')
    ECNpktsList = paraList[4].split(',')
    srcEdgeNumList = paraList[5].split(',') # being depretated
    utilLossList = paraList[6].split(',') # being depretated ?????? loss hasnt been used and may has nan bug
    maxUtil = float(paraList[7])
    netUtilList = paraList[8].split(',')#get edge utilization of the whole network. added in 2018.7.19
    sessPathUtilList = paraList[9].split(',')#get edge utilization of each path each session. added in 2018.7.20
    sessUtilList = paraList[10].split(',')#get edge utilization of each edge. added in 2018.7.20
    print("maxUtil:%s" %maxUtil)

    sessionNum = len(pacNosList)
    pacNos = []
    dels = []
    thrs = []
    ECNpkts = [] # for multi-agent the dimention is 3, otherwise, 2
    srcEdgeUL = {}
    path_util = [] # added at 7.20
    sess_util = [] # added at 7.20
    maxpath_util = [] # added at 7.21
    maxsess_util = [] # added at 7.23
    

    #get edge utilization of each path each session. added in 2018.7.20
    sessPathUtil = [] # util for each edge of each path of each session: shape:[[[...], ...], ...]
    for i in range(len(sessPathUtilList)):
        sessPathUtil.append([])
        pathUtilList = sessPathUtilList[i].split(' ')
        for j in range(len(pathUtilList)):
            sessPathUtil[i].append([])
            edgeUtilList = pathUtilList[j].split('-')
            for k in range(len(edgeUtilList)):
                sessPathUtil[i][j].append(int(edgeUtilList[k])/10000)
    #print("----->>>>>> sessPathUtil")
    #print(sessPathUtil)
    #get edge utilization of each edge. added in 2018.7.20
    sessUtil = [] # util for each path (sum util) of each session: shape:[[...], ...]
    for i in range(len(sessUtilList)):
        sessUtil.append([])
        edgeUtilList = sessUtilList[i].split(' ')
        for j in range(len(edgeUtilList)):
            sessUtil[i].append(int(edgeUtilList[j])/10000)
    #print("----->>>>>> sessUtil")
    #print(sessUtil)

    ##get edge utilization of the whole network. added in 2018.7.19
    netUtil = [] # length = EDGE_NUM; deprecated on 7.20
    netUtilLen = len(netUtilList);
    for i in range(netUtilLen):
        netUtil.append(int(netUtilList[i])/10000)
    #print("----->>>>>>>>>>>")
    #print(netUtil)

    # information of linked Edge
    k = 0
    for i in range(len(srcEdgeNumList)):
        pair_ = srcEdgeNumList[i].split(' ')
        srcNode = int(pair_[0])
        srcEdges = int(pair_[1])
        srcEdgeUL[srcNode] = []
        for j in range(srcEdges):
            util = float(utilLossList[k].split(' ')[0])
            loss = float(utilLossList[k].split(' ')[1])
            srcEdgeUL[srcNode].append([util, loss])
            k += 1
    #print(srcEdgeUL)

    if AGENT_TYPE == "multi_agent":
        # parse the information for each src Node
        for i in range(AGENT_NUM):
            pacNos.append([])
            dels.append([])
            thrs.append([])
            ECNpkts.append([])
            
            path_util.append([])
            sess_util.append([])
            maxpath_util.append([])
            maxsess_util.append([])

        for i in range(sessionNum):
            pacNosItem = list(map(int, pacNosList[i].split(' ')))
            delsItem = list(map(float, delsList[i].split(' ')))
            thrsItem = list(map(float, thrsList[i].split(' ')))
            ECNpktsItem = list(map(int, ECNpktsList[i].split(' ')))
            #print("delsItem:", delsItem)
            for j in range(len(delsItem)):
                delsItem[j] *= 1000
            pacNos[sessionSrc[i]].append(pacNosItem)
            dels[sessionSrc[i]].append(delsItem)
            thrs[sessionSrc[i]].append(thrsItem)
            ECNpkts[sessionSrc[i]].append(ECNpktsItem)
           
            # calculate path Utils or edgeUtils for each agent
            sess_util[sessionSrc[i]] += sessUtil[i]
            #sess_util[sessionSrc[i]].append(sum(sessUtil[i]))
            temp_sessmax = 0.
            for j in sessPathUtil[i]:
                #path_util[sessionSrc[i]] += j
                path_util[sessionSrc[i]].append(sum(j))
                path_util[sessionSrc[i]].append(max(j))
                maxpath_util[sessionSrc[i]].append(max(j))
                temp_sessmax = max(temp_sessmax, max(j))
            maxsess_util[sessionSrc[i]].append(temp_sessmax)
            
        #print("pacNos:", pacNos)
        #print("dels:", dels)
        #print("thrs:", thrs)
        #print("ECNpkts", ECNpkts)
        
        multi_state_new = []
        multi_reward = []
        multi_thr = []
        multi_delay = []
        multi_maxutil = []
        
        # calculate state_new, thr, delay, state and reward for each agent
        for i in range(AGENT_NUM):
            if(agents[i] == None):
                state_new = None
                reward = None
                thr = None
                delay = None
                ecnpkt = None
            else:
                thr = []
                thr_sum = []
                delay = []
                delay_sum = []
                ecnpkt = []
                for j in range(srcSessNum[i]):
                    thr += thrs[i][j]
                    thr_sum.append(sum(thrs[i][j]))
                    delay += dels[i][j]
                    delay_sum.append(sum(np.array(dels[i][j]) * np.array(pacNos[i][j])) / (sum(pacNos[i][j]) + 1e-5))
                    ecnpkt.append(sum(ECNpkts[i][j]) / (sum(pacNos[i][j]) + 1e-5))
                #thr = np.sum(np.array(thrs[i], dtype=np.float64), axis=1)
                #delay = np.sum(np.array(dels[i], dtype=np.float64), axis=1)
                
                reward = 0.
                if RWD_SELE[0] == "1":
                    #reward += np.sum(np.log(np.array(thr_sum) + 1e-5)) # temporarily replaced by cooperate reward
                    reward += np.mean(np.log(1 - np.clip(np.array(maxsess_util[i]), 0., 1.) + 1e-50)) + 0.1 * np.log(1 - np.clip(maxUtil, 0., 1.) + 1e-50)
                if RWD_SELE[1] == "1":
                    #reward += np.sum(-DELTA * np.log(np.array(delay_sum) + 1e-5)) #temporarily replaced by cooperated reward
                    #reward += np.mean(np.log(1 - np.clip(np.max(np.array(sess_util[i])), 0., 1.) + 1e-50)) + 10 * np.log(1 - np.clip(maxUtil, 0., 1.) + 1e-50)
                    #reward += np.mean(np.log(1 - np.clip(np.array(maxsess_util[i]), 0., 1.) + 1e-50)) + 0.05 * np.log(1 - np.clip(maxUtil, 0., 1.) + 1e-50) # 8-14
                    #reward += np.mean(np.log(1 - np.clip(np.array(maxsess_util[i]), 0., 1.) + 1e-50)) + 10.0 / len(maxsess_util[i]) * np.log(1 - np.clip(maxUtil, 0., 1.) + 1e-50) # 8-18
                    reward += np.mean(- 1.0 / (1 - np.clip(np.array(maxsess_util[i]), 0., 1.) + 1e-50)) + 0.1 / len(maxsess_util[i]) * (- 1.0) / (1 - np.clip(maxUtil, 0., 1.) + 1e-50)
                if RWD_SELE[2] == "1":
                    #reward += np.sum(-DELTA * np.log(np.array(ecnpkt) + 1e-5)) # temporarily replaced by sessutil + maxutil rwd
                    #reward += np.mean(np.log(1 - np.clip(np.array(sess_util[i]), 0., 1.) + 1e-50)) + np.log(1 - np.clip(maxUtil, 0., 1.) + 1e-50)
                    #reward += np.mean(np.log(1 - np.clip(np.array(maxsess_util[i]), 0., 1.) + 1e-50)) + 0.5 * np.log(1 - np.clip(maxUtil, 0., 1.) + 1e-50) # 8-14
                    reward += np.mean(np.log(1 - np.clip(np.array(maxsess_util[i]), 0., 1.) + 1e-50)) + 1.0 / len(maxsess_util[i]) * np.log(1 - np.clip(maxUtil, 0., 1.) + 1e-50)
                if RWD_SELE[3] == "1":
                    #reward += np.mean(np.log(1 - np.clip(np.array(maxsess_util[i]), 0., 1.) + 1e-50)) + np.log(1 - np.clip(maxUtil, 0., 1.) + 1e-50) # 8-18
                    #reward += np.sum(np.log(1 - np.clip(np.array(sess_util[i]), 0., 1.) + 1e-50))
                    #reward += -np.sum(np.exp(5 * np.array(sess_util[i]))) # bug reward
                    # reward += -np.sum(np.exp(10 * np.array(netUtil)))    # being deprecated
                    reward += np.mean(- 1.0 / (1 - np.clip(np.array(maxsess_util[i]), 0., 1.) + 1e-5)) + 0.1 / len(maxsess_util[i]) * (- 1.0) / (1 - np.clip(maxUtil, 0., 1.) + 1e-5)
                if RWD_SELE[4] == "1":
                    #reward += np.mean(np.log(1 - np.clip(np.max(np.array(sess_util[i])), 0., 1.) + 1e-50)) + np.log(1 - np.clip(maxUtil, 0., 1.) + 1e-50) # infact we dont need np.mean() here
                    #reward += np.mean(np.log(1 - np.clip(np.array(maxpath_util[i]), 0., 1.) + 1e-50)) + 0.1 * np.log(1 - np.clip(maxUtil, 0., 1.) + 1e-50) # 8-14
                    reward += np.mean(np.log(1 - np.clip(np.array(maxsess_util[i]), 0., 1.) + 1e-50)) + 0.1 / len(maxsess_util[i]) * np.log(1 - np.clip(maxUtil, 0., 1.) + 1e-50)

                # calculate state_new for src i
                state_new = []
                if FEAT_SELE[0] == "1": 
                    state_new += delay
                if FEAT_SELE[1] == "1":
                    state_new += delay_sum
                if FEAT_SELE[2] == "1":
                    state_new += thr
                if FEAT_SELE[3] == "1":
                    state_new += thr_sum
                if FEAT_SELE[4] == "1":
                    if i in srcEdgeUL:
                        for j in srcEdgeUL[i]:
                            state_new += j[:1]
                if FEAT_SELE[5] == "1":
                    state_new += ecnpkt
                if FEAT_SELE[6] == "1":
                    # state_new += netUtil # deprecated on 7.20
                    state_new += path_util[i]
                if FEAT_SELE[7] == "1":
                    state_new += sess_util[i]
                #state_new = np.concatenate((thr, delay))

                thr = np.sum(np.array(thr_sum))
                #delay = np.mean(np.array(delay_sum))
                delay = np.sum(np.array(delay_sum))
                
                #state_new = np.concatenate((np.log(thr), np.log(delay)))
            
            multi_state_new.append(state_new)
            multi_reward.append(reward)
            multi_thr.append(thr)
            multi_delay.append(delay)
            multi_maxutil.append(maxUtil)

        print(multi_thr, file=file_thr_out)
        print(multi_delay, file=file_del_out)
        print(maxUtil, file=file_util_out)
        return multi_state_new, multi_reward, multi_thr, multi_delay, multi_maxutil
    else:
        for i in range(sessionNum):
            pacNos.append([])
            dels.append([])
            thrs.append([])
            ECNpkts.append([])
            pacNosItem = pacNosList[i].split(' ')
            delsItem = delsList[i].split(' ')
            thrsItem = thrsList[i].split(' ')
            ECNpktsItem = ECNpktsList[i].split(' ')
            pathNum = len(pacNosItem)
            for j in range(pathNum):
                pacNos[i].append(int(pacNosItem[j]))
                dels[i].append(float(delsItem[j])*1000)
                thrs[i].append(float(thrsItem[j]))
                ECNpkts[i].append(int(ECNpktsItem[j]))
    
        print("pacNos:", pacNos)
        #print("dels:", dels)
        #print("thrs:", thrs)
        #print("ECNpkts", ECNpkts)

        # calculate the sum thrs and dels for each session
        thrs_ = []
        dels_ = []
        ECNpkts_ = []
        for i in range(sessionNum):
            thrs_.append(sum(thrs[i]))
            dels_.append(sum(np.array(dels[i]) * np.array(pacNos[i])) / (sum(pacNos[i]) + 1e-5))
            ECNpkts_.append(float(sum(ECNpkts[i]))/ (float(sum(pacNos[i])) + 1e-5))

        thr = np.array(thrs_, dtype=np.float64)
        delay = np.array(dels_, dtype=np.float64)
        ecnpkt = np.array(ECNpkts_, dtype=np.float64)

        #reward = np.sum(np.log(thr) - DELTA * np.log(delay))
        reward = -np.sum(np.exp(np.array(netUtil) + 1))
        #state_new = np.concatenate((thr, delay))
        state_new = np.array(netUtil)

        print(list(thr), file=file_thr_out)
        print(list(delay), file=file_del_out)
        print(maxUtil, file=file_util_out)
        return [state_new], [reward], [np.sum(thr)], [np.mean(delay)], [maxUtil]

def step(tmp):
    global ret_c
    state, reward, thr, dly, maxutil = split_arg(tmp)
    print("state:", state)
    print("reward:", reward)
    print("thr:", thr)
    print("dly:", dly)

    #if not np.all(state) or not np.all(reward):
    #    print('invalid...')
        # ret_c = tuple(utilize.get_rnd_solution(DIM_ACTION, NUM_PATHS))
    #    ret_c = tuple(action)
    #    return ret_c
    if AGENT_TYPE == "multi_agent":
        ret_c_t = []
        ret_c = []
        for i in range(AGENT_NUM):
            if agents[i] != None:
                #print("state:", state)
                #print("predict:", i, state[i], reward[i], thr[i], dly[i])
                print("state:", i, state[i])
                print("reward:", reward[i])
                ret_c_t.append(agents[i].predict(state[i], reward[i], thr[i], dly[i], maxutil[i]))
            else:
                ret_c_t.append([])
        for i in range(len(sessionSrc)):
            ret_c += ret_c_t[sessionSrc[i]][0:NUM_PATHS[i]]
            ret_c_t[sessionSrc[i]] = ret_c_t[sessionSrc[i]][NUM_PATHS[i]:]
    elif AGENT_TYPE == "drl_te":
        ret_c = tuple(agents[0].predict(state[0], reward[0], thr[0], dly[0], maxutil[0]))
    else:
        # for OSPF and MCF
        ret_c = action

    print("action:", ret_c)
    # print('rwd', reward)
    # print('act', ret)
    # print('act_c', ret_c)
    # print(agent.timer)

    reward_t = 0.
    for i in reward:
        if i != None:
            reward_t += i # for multiagent each agent has a reward, otherwise there is only one agent

    print(state, file=file_sta_out)
    print(reward_t, file=file_rwd_out)
    print(ret_c, file=file_act_out)

    return ret_c

'''
lcy: a function that not being used?
'''
def sim_ddpg():
    # print("come into sim_ddpg")
    env = Env(DIM_STATE, DIM_ACTION, SEED, NUM_PATHS)
    state = env.state_init
    action = utilize.convert_action(env.state_init, NUM_PATHS)
    agent = DrlAgent(state, action)
    print('Best Solution:', env.best_sol)
    for ep in range(MAX_EPISODES):
        for ep in range(MAX_EP_STEPS):
            sn, r = env.getReward(state, action)
            action = agent.predict(sn, r)
            state = sn
            # print(state)
        #state = action = env.reset()


if __name__ == "__main__":
    print("drlte ----------------------")
    msgTotalLen = 0
    msgRecvLen = 0
    msg = ""
    while True:
        datarecv = tcpSocket.recv(BUFSIZE).decode()
        if len(datarecv) > 0:
            if msgTotalLen == 0:
                totalLenStr = (datarecv.split(';'))[0]
                print(totalLenStr, totalLenStr)
                msgTotalLen = int(totalLenStr) + len(totalLenStr) + 1#1 is the length of ';'
                if msgTotalLen == 2:#stop signal
                    print("simulation is over!")
                    break;
            msgRecvLen += len(datarecv)
            msg += datarecv
            if msgRecvLen < msgTotalLen: 
                continue
            
            #print("MSG:", msg)
            
            ret_c = step(msg)
            msg = ""
            for i in range(len(ret_c)-1):
                msg += str(round(ret_c[i], 3)) + ','
            msg += str(round(ret_c[len(ret_c)-1], 3))
            msg = str(len(msg)) + ';' + msg;
            
            #print("retMSG:", msg)

            msgTotalLen = len(msg)
            blockNum = int((msgTotalLen+blockSize-1)/blockSize);
            for i in range(blockNum):
                data = msg[i*blockSize:i*blockSize+blockSize]
                tcpSocket.send(data.encode())
            msgTotalLen = 0
            msgRecvLen = 0
            msg = ""
    tcpSocket.close()

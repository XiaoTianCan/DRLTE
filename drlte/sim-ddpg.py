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

DIR_SUM = getattr(FLAGS, 'dir_sum').format(TIME_STAMP)
DIR_RAW = getattr(FLAGS, 'dir_raw').format(TIME_STAMP)
DIR_MOD = getattr(FLAGS, 'dir_mod').format(TIME_STAMP)
DIR_LOG = getattr(FLAGS, 'dir_log').format(TIME_STAMP)
os.mkdir(DIR_LOG)

class DrlAgent:
    def __init__(self, state_init, action_init):
        sess = tf.Session()

        self.__actor = ActorNetwork(sess, DIM_STATE, DIM_ACTION, ACTION_BOUND,
                                    ACTOR_LEARNING_RATE, TAU, NUM_PATHS)
        self.__critic = CriticNetwork(sess, DIM_STATE, DIM_ACTION,
                                      CRITIC_LEARNING_RATE, TAU, self.__actor.num_trainable_vars)

        self.__prioritized_replay = PrioritizedReplayBuffer(BUFFER_SIZE, MINI_BATCH, ALPHA, MU, SEED)
        self.__replay = ReplayBuffer(BUFFER_SIZE, SEED)

        self.__summary = Summary(sess, DIR_SUM)
        self.__summary.add_variable(name='throughput')
        self.__summary.add_variable(name='delay')
        self.__summary.add_variable(name='reward')
        self.__summary.add_variable(name='ep-reward')
        self.__summary.add_variable(name='ep-max-q')
        self.__summary.build()
        #self.__summary.write_vars(FLAGS)

        self.__explorer = Explorer(EP_BEGIN, EP_END, EP_ST, DIM_ACTION, NUM_PATHS, SEED)

        sess.run(tf.global_variables_initializer())
        self.__actor.update_target_paras()
        self.__critic.update_target_paras()

        self.__state_curt = state_init
        self.__action_curt = action_init
        self.__base_sol = utilize.get_base_solution(DIM_ACTION)

        self.__episode = 0
        self.__step = 0
        self.__ep_reward = 0.
        self.__ep_avg_max_q = 0.

        self.__beta = BETA

    @property
    def timer(self):
        return '| %s '%ACT_FLAG \
               + '| tm: %s '%datetime.datetime.now() \
               + '| ep: %.4d '%self.__episode \
               + '| st: %.4d '%self.__step

    def predict(self, state_new, reward, thr, dly):
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
              '| action: ' + '%.2f ' * DIM_ACTION % tuple(self.__action_curt))
        if self.__step >= MAX_EP_STEPS:
            self.__step = 0
            self.__episode += 1
            self.__ep_reward = 0.
            self.__ep_avg_max_q = 0.

        action_original = self.__actor.predict([state_new])[0]

        # print('act_o', action_original)

        action = self.__explorer.get_act(action_original, self.__episode, flag=ACT_FLAG)

        # print('act_s', action)

        # Priority
        target_q = self.__critic.predict_target(
            [state_new], self.__actor.predict_target([state_new]))[0]
        value_q = self.__critic.predict([self.__state_curt], [self.__action_curt])[0]
        grads = self.__critic.calculate_gradients([self.__state_curt], [self.__action_curt])
        td_error = abs(reward + GAMMA * target_q - value_q)

        transition = (self.__state_curt, self.__action_curt, reward, state_new)
        self.__prioritized_replay.add(transition, td_error, abs(np.mean(grads[0])))
        self.__replay.add(transition[0], transition[1], transition[2], transition[3])

        self.__state_curt = state_new
        self.__action_curt = action

        if len(self.__prioritized_replay) > MINI_BATCH:
            self.train()

        return action

    def train(self):
        self.__beta += (1-self.__beta) / EP_ST
        batch, weights, indices = self.__prioritized_replay.select(self.__beta)
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

        predicted_q, _ = self.__critic.train(batch_state, batch_action, batch_y, weights)

        self.__ep_avg_max_q += np.amax(predicted_q)

        a_outs = self.__actor.predict(batch_state)
        grads = self.__critic.calculate_gradients(batch_state, a_outs)

        print('*'*20)
        print(grads[0])

        # Prioritized
        self.__prioritized_replay.priority_update(indices,
                                                  np.array(batch_error).flatten(),
                                                  abs(np.mean(grads[0], axis=1)))

        self.__actor.train(batch_state, grads[0])

        self.__actor.update_target_paras()
        self.__critic.update_target_paras()


if not SIM_FLAG:
    ns3Server = (SERVER_IP, SERVER_PORT)
    tcpSocket = socket(AF_INET, SOCK_STREAM)
    tcpSocket.connect(ns3Server)

    msgTotalLen = 0
    msgRecvLen = 0
    msg = ""
    blockSize = 1024;
    BUFSIZE = 1025
    sessionSrc = []
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
            ###print(msg)
            msgList = (msg.split(';')[1]).split(',');
            DIM_STATE = len(msgList)#sessionNum*2
            DIM_ACTION = 0
            NUM_PATHS = []
            for i in range(len(msgList)):
                if i%2 == 0:
                    sessionSrc.append(int(msgList[i]))
                else:
                    DIM_ACTION += int(msgList[i])
                    NUM_PATHS.append(int(msgList[i]))
            break


    state = np.zeros(DIM_STATE)
    action = utilize.convert_action(np.ones(DIM_ACTION), NUM_PATHS)

    agent = DrlAgent(state, action)
    ret_c = tuple(action)

    file_sta_out = open(DIR_LOG + '/sta.log', 'a', 1)
    file_rwd_out = open(DIR_LOG + '/rwd.log', 'a', 1)
    file_act_out = open(DIR_LOG + '/act.log', 'a', 1)
    file_thr_out = open(DIR_LOG + '/thr.log', 'a', 1)
    file_del_out = open(DIR_LOG + '/del.log', 'a', 1)
'''
some notes:
sessionSrc = [src_of_s1, src_of_s2, ...]#this is a list of source node of each session; its length is the number of sessions
pacNos = [[],[],[],...]#a list of packet number for each session and each path; the list has two dimensions: session and path
dels = [[],[],[],...]#similar to pacNos, except that the content is time delays for each session and each path
thrs = [[],[],[],...]#tong shang
ECNpkts = [[],[],[],...]#tong shang

When do distributed drl learning, multiple agent=DrlAgent(..) are needed, so the upper codes need to be modified
the state of distributed drl learning also contains two kinds of parameters: delay and throughput #we may add, delete, replace parameters in the future
'''
def split_arg(para):
    paraList = para.split(';')
    pacNosList = paraList[1].split(',')
    delsList = paraList[2].split(',')
    thrsList = paraList[3].split(',')
    ECNpktsList = paraList[4].split(',')
    sessionNum = len(pacNosList)
    pacNos = []
    dels = []
    thrs = []
    ECNpkts = []
    #print(pacNosList)
    #print(delsList)
    #print(thrsList)
    #print(ECNpktsList)

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

    thr = np.sum(np.array(thrs, dtype=np.float64), axis=1)
    delay = np.sum(np.array(dels, dtype=np.float64), axis=1)

    reward = np.sum(np.log(thr) - DELTA * np.log(delay))
    state_new = np.concatenate((thr, delay))
    #state_new = np.concatenate((np.log(thr), np.log(delay)))

    print(thr, file=file_thr_out)
    print(delay, file=file_del_out)
    return state_new, reward, np.sum(thr), np.mean(delay)

def step(tmp):
    global ret_c
    state, reward, thr, dly = split_arg(tmp)

    if not np.all(state) or not np.all(reward):
        print('invalid...')
        # ret_c = tuple(utilize.get_rnd_solution(DIM_ACTION, NUM_PATHS))
        ret_c = tuple(action)
        return ret_c

    ret_c = tuple(agent.predict(state, reward, thr, dly))

    # print('rwd', reward)
    # print('act', ret)
    # print('act_c', ret_c)
    # print(agent.timer)

    print(state, file=file_sta_out)
    print(reward, file=file_rwd_out)
    print(ret_c, file=file_act_out)

    return ret_c


def sim_ddpg():
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
                msgTotalLen = int(totalLenStr) + len(totalLenStr) + 1#1 is the length of ';'
                if msgTotalLen == 2:#stop signal
                    print("simulation is over!")
                    break;
            msgRecvLen += len(datarecv)
            msg += datarecv
            if msgRecvLen < msgTotalLen: 
                continue
            ###print(msg)
            ret_c = step(msg)
            msg = ""
            for i in range(len(ret_c)-1):
                msg += str(round(ret_c[i], 3)) + ','
            msg += str(round(ret_c[len(ret_c)-1], 3))
            msg = str(len(msg)) + ';' + msg;
            msgTotalLen = len(msg)
            blockNum = int((msgTotalLen+blockSize-1)/blockSize);
            for i in range(blockNum):
                data = msg[i*blockSize:i*blockSize+blockSize]
                tcpSocket.send(data.encode())
            msgTotalLen = 0
            msgRecvLen = 0
            msg = ""
    tcpSocket.close()
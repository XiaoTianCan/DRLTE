import tensorflow as tf
import os
from os.path import join as pjoin


def home_out(path):
    full_path = pjoin('/home', 'netlab', 'gengnan', 'drl_te', path)
    #print(full_path)
    if not os.path.exists(full_path):
        os.mkdir(full_path)
    return full_path

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('server_port', '50001', 'remote server port')
flags.DEFINE_string('server_ip', '127.0.0.1', 'remote server ip address')

flags.DEFINE_string('act_flag', 'drl', 'methods for explorer')
flags.DEFINE_boolean('sim_flag', False, 'simulation flag')
flags.DEFINE_integer('random_seed', 66, "seed for random generation")

flags.DEFINE_integer('size_buffer', 100000, 'the size of replay buffer')
flags.DEFINE_integer('mini_batch', 32, "size of mini batch")

flags.DEFINE_float('action_bound', 1., 'max action')

flags.DEFINE_integer('episodes', 600, "training episode")
flags.DEFINE_integer('epochs', 30, 'training epochs for each episode')

flags.DEFINE_float('epsilon_begin', 1., "the begin value of epsilon random")
flags.DEFINE_float('epsilon_end', 0., "the end value of epsilon random")
flags.DEFINE_integer('epsilon_steps', 1200, "the steps for epsilon random")

flags.DEFINE_float('learning_rate_actor', 0.0001, "learning rate for actor network")
flags.DEFINE_float('learning_rate_critic', 0.001, "learning rate for critic network")

flags.DEFINE_float('gamma', 0.99, "discount value for reward")
flags.DEFINE_float('alpha', 0.6, 'prioritized replay buffer parameter alpha')
flags.DEFINE_float('beta', 0.5, 'prioritized replay buffer parameter IS')
flags.DEFINE_float('mu', 0.6, 'Prioritized replay buffer parameter DDPG')
flags.DEFINE_float('tau', 0.001, "tau for target network update")
flags.DEFINE_float('delta', 0.1, 'trade off throughput and delay')

flags.DEFINE_string('dir_sum', home_out('sum') + '/{0}', "the path of tf summary")
flags.DEFINE_string('dir_raw', home_out('raw') + '/{0}', 'the path of raw data')
flags.DEFINE_string('dir_mod', home_out('mod') + '/{0}', 'the path of saved models')
flags.DEFINE_string('dir_log', home_out('log') + '/{0}', 'the path of logs')
flags.DEFINE_string('dir_ckpoint', home_out('ckpoint') + '/{0}', 'the path of checkpoints')

# flags for features selection
flags.DEFINE_string("feature_select", "00000001", "from left to right: delay(path and sess), thr_per_path, thr_per_sess, linked edge info, ecn, utils for each edge of each path of each session, utils for each edge of each session(uniqued)")
flags.DEFINE_string("stamp_type", "__TIME_STAMP", "the stamp style for the name of log directory, may be TIME_STAMP or other specifial dir name show the features of the expr")
flags.DEFINE_string("reward_type", "00001", "the type of the reward, from left to right, thr, delay, ecn, sess_maxutils+global_maxutils, agent_maxutils+global_maxutils")
flags.DEFINE_string("agent_type", "multi_agent", "the method type for the agent include: drl_te, OSPF, MCF, OBL or multi_agent") 
flags.DEFINE_string("mcf_path", None, "the answer files path for mcf method")
flags.DEFINE_string("obl_path", None, "the answer files path for obl method")
flags.DEFINE_string("or_path", None, "the answer files path for OR method")
flags.DEFINE_string("action_path", None, "the original action for the explorer"); # modified by lcy: 2018-9-1 

# flags for dynamic learning rate (for multiagent)
flags.DEFINE_float("deta_w", 1., "the ordinary learning rate factor")
flags.DEFINE_float("deta_l", 1., "the learning rate factor for the bad situation") # deta_l > deta_w for multiagent

# flags for exploration episodes
flags.DEFINE_integer("explore_epochs", 0, "the explore times before the final exploration")
flags.DEFINE_integer("explore_decay", 300, "the learning decay step for the pre-exploration")

# flags for para reload
flags.DEFINE_string("ckpt_path", None, "the checkpoint path, None means not using checkpoints to initialize the paramaters")

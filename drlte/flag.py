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

flags.DEFINE_string('act_flag', 'drl', 'methods for explorer')
flags.DEFINE_boolean('sim_flag', False, 'simulation flag')
flags.DEFINE_integer('random_seed', 66, "seed for random generation")

flags.DEFINE_integer('dim_state', 20*2, 'the dimension of state')
flags.DEFINE_integer('dim_action', 20*3, 'the dimension of action')

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

flags.DEFINE_float('gamma', 0.01, "discount value for reward")
flags.DEFINE_float('alpha', 0.6, 'prioritized replay buffer parameter alpha')
flags.DEFINE_float('beta', 0.5, 'prioritized replay buffer parameter IS')
flags.DEFINE_float('mu', 0.6, 'Prioritized replay buffer parameter DDPG')
flags.DEFINE_float('tau', 0.001, "tau for target network update")
flags.DEFINE_float('delta', 1., 'trade off throughput and delay')

flags.DEFINE_string('dir_sum', home_out('sum') + '/{0}', "the path of tf summary")
flags.DEFINE_string('dir_raw', home_out('raw') + '/{0}', 'the path of raw data')
flags.DEFINE_string('dir_mod', home_out('mod') + '/{0}', 'the path of saved models')
flags.DEFINE_string('dir_log', home_out('log') + '/{0}', 'the path of logs')

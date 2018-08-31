"""
    Critic of Deep Deterministic policy gradient

"""
import tflearn
import tensorflow as tf

class CriticNetwork():
    def __init__(self, session, dim_state, dim_action, learning_rate, tau, num_actor_vars):
        self.__sess = session
        self.__dim_s = dim_state
        self.__dim_a = dim_action
        self.__learning_rate = learning_rate
        self.__tau = tau
        
        cur_para_num = len(tf.trainable_variables())
        self.__inputs, self.__action, self.__out = self.buildNetwork()
        #self.__paras = tf.trainable_variables()[num_actor_vars:]
        self.__paras = tf.trainable_variables()[cur_para_num:]

        self.__target_inputs, self.__target_action, self.__target_out = self.buildNetwork()
        #self.__target_paras = tf.trainable_variables()[(len(self.__paras) + num_actor_vars):]
        self.__target_paras = tf.trainable_variables()[(len(self.__paras) + cur_para_num):]

        self.__ops_update_target = []
        for i in range(len(self.__target_paras)):
            val = tf.add(tf.multiply(self.__paras[i], self.__tau), tf.multiply(self.__target_paras[i], 1. - self.__tau))
            op = self.__target_paras[i].assign(val)
            self.__ops_update_target.append(op)

        self.__q_predicted = tf.placeholder(tf.float32, [None, 1])
        self.__is_weight = tf.placeholder(tf.float32, [None, 1])

        self.loss = tflearn.mean_square(self.__q_predicted, self.__out)
        self.loss = tf.multiply(self.loss, self.__is_weight)
        self.optimize = tf.train.AdamOptimizer(self.__learning_rate).minimize(self.loss)

        self.__gradient_action = tf.gradients(self.__out, self.__action)
        # self.__gradient_action = tf.div(tf.gradients(self.__out, self.__action),
        #             tf.constant(MINIBATCH_SIZE, dtype=tf.float32))

    def buildNetwork(self):
        inputs = tflearn.input_data(shape=[None, self.__dim_s])
        action = tflearn.input_data(shape=[None, self.__dim_a])

        # net = tflearn.batch_normalization(inputs)
        net = inputs
        
        # temp modified by lcy
        ##net = tflearn.fully_connected(net, 128, activation='LeakyReLU')
        # end modified
        
        net = tflearn.fully_connected(net, 32, activation='LeakyReLU')

        layer1 = tflearn.fully_connected(net, 64)
        layer2 = tflearn.fully_connected(action, 64)

        tmp = tf.matmul(net, layer1.W) + tf.matmul(action, layer2.W) + layer2.b

        net = tflearn.activation(tmp, activation='LeakyReLU')

        w_init = tflearn.initializations.uniform(minval=-3e-3, maxval=3e-3)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)

        return inputs, action, out

    def train(self, inputs, action, q_predicted, is_weight):
        return self.__sess.run([self.__out, self.optimize], feed_dict={
            self.__inputs: inputs,
            self.__action: action,
            self.__q_predicted: q_predicted,
            self.__is_weight: is_weight
        })

    def predict(self, inputs, action):
        return self.__sess.run(self.__out, feed_dict={
            self.__inputs: inputs,
            self.__action: action
        })

    def predict_target(self, inputs, action):
        return self.__sess.run(self.__target_out, feed_dict={
            self.__target_inputs: inputs,
            self.__target_action: action
        })

    def calculate_gradients(self, inputs, action):
        return self.__sess.run(self.__gradient_action, feed_dict={
            self.__inputs: inputs,
            self.__action: action
        })

    def update_target_paras(self):
        self.__sess.run(self.__ops_update_target)

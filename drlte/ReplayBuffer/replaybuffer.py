"""

	Replay Buffer for Deep Reinforcement Learning

"""

from collections import deque
import random
import numpy as np

from ReplayBuffer import sum_tree


class ReplayBuffer():
    def __init__(self, size_buffer, random_seed=8):
        self.__size_bf = size_buffer
        self.__length = 0
        self.__buffer = deque()
        random.seed(random_seed)
        np.random.seed(random_seed)


    @property
    def buffer(self):
        return self.__buffer


    def add(self, state, action, reward, state_next):
        exp = (state, action, reward, state_next)
        if self.__length < self.__size_bf:
            self.__buffer.append(exp)
            self.__length += 1
        else:
            self.__buffer.popleft()
            self.__buffer.append(exp)

    def add_batch(self, batch_s, batch_a, batch_r, batch_sn):
        for i in range(len(batch_s)):
            self.add(batch_s[i], batch_a[i], batch_r[i], batch_sn[i])

    def __len__(self):
        return self.__length

    def sample_batch(self, size_batch):

        if self.__length < size_batch:
            batch = random.sample(self.__buffer, self.__length)
        else:
            batch = random.sample(self.__buffer, size_batch)

        batch_s = np.array([d[0] for d in batch])
        batch_a = np.array([d[1] for d in batch])
        batch_r = np.array([d[2] for d in batch])
        batch_sn = np.array([d[3] for d in batch])

        return batch_s, batch_a, batch_r, batch_sn

    def clear(self):
        self.__buffer.clear()
        self.count = 0


class PrioritizedReplayBuffer():
    def __init__(self, memory_size, batch_size, alpha, mu, seed):
        """ Prioritized experience replay buffer initialization.

        Parameters
        ----------
        memory_size : int
            sample size to be stored
        batch_size : int
            batch size to be selected by `select` method
        alpha: float
            exponent determine how much prioritization.
            Prob_i \sim priority_i**alpha/sum(priority**alpha)
        """
        self.tree = sum_tree.SumTree(memory_size)
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.__e = 0.01
        self.__mu = mu
        np.random.seed(seed)

    def __len__(self):
        return self.tree.filled_size()

    def add(self, data, error, gradient):
        """ Add new sample.

        Parameters
        ----------
        data : object
            new sample
        error : float
            sample's td-error
        """
        priority = self.__getPriority(error, gradient)
        self.tree.add(data, priority)

    def __getPriority(self, error, gradient):
        priority = self.__mu * np.array(error) + (1 - self.__mu) * np.array(gradient)
        idx = np.where(priority<0.)
        if len(idx[0]) != 0:
            for i in idx[0]:
                priority[i] = 0.
        return (priority + self.__e) ** self.alpha

    def select(self, beta):
        """ The method return samples randomly.

        Parameters
        ----------
        beta : float

        Returns
        -------
        out :
            list of samples
        weights:
            list of weight
        indices:
            list of sample indices
            The indices indicate sample positions in a sum tree.
        """

        if self.tree.filled_size() < self.batch_size:
            print('LESS and LESS')
            return None, None, None

        out = []
        indices = []
        weights = []

        segment = self.tree.root / self.batch_size

        for i in range(self.batch_size):
            min_val = segment * i
            max_val = segment * (i + 1)
            r = random.uniform(min_val, max_val)
            data, priority, index = self.tree.find(r, norm=False)

            weights.append((1. / self.memory_size / priority) ** beta if priority > 1e-16 else 0)
            indices.append(index)
            out.append(data)

        weights /= max(weights)  # Normalize for stability

        return out, weights, indices

    def priority_update(self, indices, error, gradient):
        """ The methods update samples's priority.

        Parameters
        ----------
        indices :
            list of sample indices
        """
        priorities = self.__getPriority(error, gradient)
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, p)

    def reset_alpha(self, alpha):
        """ Reset a exponent alpha.
        Parameters
        ----------
        alpha : float
        """
        self.alpha, old_alpha = alpha, self.alpha
        priorities = [(self.tree.get_val(i) + self.__e) ** -old_alpha for i in range(self.tree.filled_size())]
        self.priority_update(range(self.tree.filled_size()), priorities)




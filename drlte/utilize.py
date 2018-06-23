import numpy as np


def aggregate(paths, num_paths):
    act = np.split(paths, np.cumsum(num_paths))[:-1]
    ret = [np.mean(i) for i in act]
    return np.array(ret)


def softmax(x, num_paths):
    act = np.split(x, np.cumsum(num_paths))[:-1]
    for idx, val in enumerate(act):
        act[idx] = np.exp(val) / np.sum(np.exp(val), axis=0)
    ret = []
    for ary in act:
        for val in ary.flatten():
            ret.append(val.item())
    return ret


def convert_action(action, num_paths):
    act = np.split(action, np.cumsum(num_paths))[:-1]
    for idx, val in enumerate(act):
        if not np.any(val):
            act[idx][0] = 1.
            continue
        act[idx] = val / sum(val.flatten())
    ret = []
    for ary in act:
        for val in ary.flatten():
            ret.append(val.item())
    return ret


def get_base_solution(dim_action):
    return [0,0,20,0,16,0,0,0,6,0,19,0,0,0,7,0,14,0,20,0,0,0,20,0,0,20,0,0,15,0,0,0,20,0,0,12,0,0,20,0,0,12,0,19,0,0,0,20,0,0,20,20,0,0,0,0,10,6,0,0]


def get_fix_solution():
    return [0.9998,9.99803e-05,9.99803e-05,0.167924,0.000117319,0.831958,8.16729e-05,0.183207,0.816712,0.820988,
            0.000114161,0.178897,0.927965,0.0719427,9.27968e-05,0.0871434,0.000121083,0.912736,9.99823e-05,9.99823e-05,
            0.9998,0.999509,0.000410313,8.09819e-05,0.792469,0.207451,7.92509e-05,0.838372,0.000111352,0.161517,
            7.84816e-05,0.999843,7.84816e-05,0.996286,7.99973e-05,0.00363412,8.02695e-05,0.999839,8.02695e-05,0.11981,
            0.880102,8.80102e-05,8.35498e-05,0.16442,0.835496,0.999833,8.3323e-05,8.3323e-05,0.9998,9.99802e-05,
            9.99802e-05,0.0243314,0.782951,0.192717,7.61769e-05,0.999848,7.61769e-05,0.0175442,7.40197e-05,0.982382]


def get_ext_solution(dim_action, num_paths):
    tmp = []
    for pt in num_paths:
        a = np.zeros(pt)
        a[0] = 1
        a = np.random.permutation(a)
        tmp += a.tolist()
    return convert_action(tmp, num_paths)


def get_rnd_solution(dim_action, num_paths):
    tmp = np.random.random(dim_action)
    return convert_action(tmp, num_paths)


def run_action(base_sol, action):
    tmp = np.array(base_sol) + np.array(action)
    tmp[tmp < 0.] = 0.000001
    # tmp[tmp > 10.] = 10.
    return tmp

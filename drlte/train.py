#coding=utf-8
import os
import sys

target = sys.argv[1]

session_nums = ['10', '20', '30', '40', '50', '60']
#session_nums = ['30']
graph_types = ['NSF', '3967'] 
demands = ['50', '70', '90', '110', '130', '150']

if target == "multi_agent":
    methods = ['OBL_3']
    state_reward = [["00000001", "00001"]]
    #state_reward = [["00000001", "00010"]] 
    #state_reward = [["00000001", "00100"]]
    #state_reward = [["00000001", "01000"]]
    #state_reward = [["00000001", "10000"]]

    for graph in graph_types:
        for sess_num in session_nums:
            for dema in demands:
                for meth in methods:
                    for i in state_reward:
                        filename = graph + "_" + sess_num + "_" + meth + "_" + dema + "b";
                        if not os.path.exists("../inputs/" + filename + ".txt"):
                            continue
                        dir_name = '_'.join(["sdemand_multi_agent", graph, sess_num, dema, meth, i[0], i[1]])
                        if os.path.exists("/home/netlab/gengnan/drl_te/log/" + dir_name):
                            continue
                        print dir_name
                        #act_path = "../inputs/OBLsolution/" + "_".join(["OBL", graph, sess_num, dema]) + ".txt"
                        ind = 0
                        while(ind < 100):
                            ind += 1 # to make the program can be stoped
                            ret = os.system("python3 sim-ddpg.py --server_port=50053 --epochs=1500 --agent_type=multi_agent --epsilon_begin=0.8 --feature_select=" + i[0] + " --reward_type=" + i[1] + " --stamp_type=" + dir_name + " --deta_w=1. --deta_l=10.  --epsilon_steps=1200 --explore_epochs=3 --explore_decay=300") # for dynamic learning rate of multiagent
                            print "result:", ret
                            if ret != 256:
                                break
elif target == "drlte":
    methods = ['OBL_3']
    for graph in graph_types:
        for sess_num in session_nums:
            for dema in demands:
                for meth in methods:
                    filename = graph + "_" + sess_num + "_" + meth + "_" + dema;
                    if not os.path.exists("../inputs/" + filename + ".txt"):
                        continue
                    dir_name = '_'.join(["drlte", graph, sess_num, dema, meth])
                    if os.path.exists("/home/netlab/gengnan/drl_te/log/" + dir_name):
                        continue
                    print dir_name
                    ind = 0
                    while(ind < 100):
                        ind += 1
                        ret = os.system("python3 sim-ddpg.py --server_port=50012 --epochs=2000  --agent_type=drl_te --epsilon_begin=0.5" + " --stamp_type=" + dir_name)
                        print "result:", ret
                        if ret != 256:
                            break
elif target == "OSPF":
    methods = ['SHR_1']
    for graph in graph_types:
        for sess_num in session_nums:
            for dema in demands:
                for meth in methods:
                    filename = graph + "_" + sess_num + "_" + meth + "_" + dema + "b";
                    if not os.path.exists("../inputs/" + filename + ".txt"):
                        continue
                    dir_name = '_'.join(["OSPFb", graph, sess_num, dema, meth])
                    if os.path.exists("/home/netlab/gengnan/drl_te/log/" + dir_name):
                        continue
                    print dir_name
                    ind = 0
                    while(ind < 100):
                        ind += 1
                        ret = os.system("python3 sim-ddpg.py --server_port=50008 --epochs=2000  --agent_type=OSPF --epsilon_begin=0.5" + " --stamp_type=" + dir_name)
                        print "result:", ret
                        if ret != 256:
                            break
elif target == "MCF":
    methods = ['MCF']
    for graph in graph_types:
        for sess_num in session_nums:
            for dema in demands:
                for meth in methods:
                    filename = graph + "_" + sess_num + "_" + meth + "_3_" + dema + "b";
                    if not os.path.exists("../inputs/" + filename + ".txt"):
                        continue
                    dir_name = '_'.join(["MCFb", graph, sess_num, dema, meth])
                    if os.path.exists("/home/netlab/gengnan/drl_te/log/" + dir_name):
                        continue
                    print dir_name
                    mcf_path = "../inputs/MCFsolution/" + "_".join(["MCF", graph, sess_num, dema]) + ".txt"
                    ind = 0
                    while(ind < 100):
                        ind += 1 
                        ret = os.system("python3 sim-ddpg.py --server_port=50010 --epochs=2000  --agent_type=MCF --epsilon_begin=0.5" + " --stamp_type=" + dir_name + " --mcf_path=" + mcf_path)
                        print "result:", ret
                        if ret != 256:
                            break
if target == "OBL":
    methods = ['OBL_3']

    for graph in graph_types:
        for sess_num in session_nums:
            for dema in demands:
                for meth in methods:
                    filename = graph + "_" + sess_num + "_" + meth + "_" + dema;
                    if not os.path.exists("../inputs/" + filename + ".txt"):
                        continue
                    dir_name = '_'.join(["OBL", graph, sess_num, dema, meth])
                    if os.path.exists("/home/netlab/gengnan/drl_te/log/" + dir_name):
                        continue
                    print dir_name
                    obl_path = "../inputs/OBLsolution/" + "_".join(["OBL", graph, sess_num, dema]) + ".txt"
                    ind = 0
                    while(ind < 100):
                        ind += 1 # to make the program can be stoped
                        ret = os.system("python3 sim-ddpg.py --server_port=50010 --epochs=1500 --agent_type=OBL --stamp_type=" + dir_name + " --obl_path=" + obl_path) 
                        print "result:", ret
                        if ret != 256:
                            break
elif target == "OR":
    methods = ['OR']
    for graph in graph_types:
        for sess_num in session_nums:
            for dema in demands:
                for meth in methods:
                    filename = graph + "_" + sess_num + "_" + meth + "_100_" + dema + "b";
                    if not os.path.exists("../inputs/" + filename + ".txt"):
                        continue
                    dir_name = '_'.join(["ORb", graph, sess_num, dema, meth])
                    if os.path.exists("/home/netlab/gengnan/drl_te/log/" + dir_name):
                        continue
                    print dir_name
                    or_path = "../inputs/ORsolution/" + "_".join(["OR", graph, sess_num, dema]) + ".txt"
                    ind = 0
                    while(ind < 100):
                        ind += 1 
                        ret = os.system("python3 sim-ddpg.py --server_port=50009 --epochs=2000  --agent_type=OR --epsilon_begin=0.5" + " --stamp_type=" + dir_name + " --or_path=" + or_path)
                        print "result:", ret
                        if ret != 256:
                            break



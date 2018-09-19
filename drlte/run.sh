#########################################################################
# File Name: run.sh
# Author: lcy
# mail:liucy15@mails.tsinghua.edu.cn
# Created Time: 2018年06月26日 星期二 19时08分12秒
#########################################################################
#!/bin/bash
python3 sim-ddpg.py --server_port=50006 --epochs=1500 --epsilon_begin=0.8 --feature_select=00000001 --reward_type=01000 --deta_w=1. --deta_l=10. --agent_type=multi_agent --mcf_path=../inputs/MCFsolution/MCF_NSF_30_50.txt --stamp_type=9-19_NSF_30_50c_OBL_3_test_gamma_smallrwd_s3_linearrwd_2300step --epsilon_steps=2300 --explore_epochs=3 --explore_decay=300 --delta=0.1

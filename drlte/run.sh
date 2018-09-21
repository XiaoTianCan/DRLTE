#########################################################################
# File Name: run.sh
# Author: lcy
# mail:liucy15@mails.tsinghua.edu.cn
# Created Time: 2018年06月26日 星期二 19时08分12秒
#########################################################################
#!/bin/bash
python3 sim-ddpg.py --server_port=50016 --epochs=1500 --epsilon_begin=0.8 --feature_select=00000001 --reward_type=01000 --deta_w=1. --deta_l=10. --agent_type=multi_agent --mcf_path=../inputs/MCFsolution/MCF_NSF_30_50.txt --stamp_type=9-21_test_saver_30_150_5global --epsilon_steps=2300 --explore_epochs=3 --explore_decay=300 --delta=5.

#########################################################################
# File Name: run.sh
# Author: lcy
# mail:liucy15@mails.tsinghua.edu.cn
# Created Time: 2018年06月26日 星期二 19时08分12秒
#########################################################################
#!/bin/bash
python3 sim-ddpg.py --server_port=50015 --epochs=2000 --epsilon_begin=0.5 --feature_select=00000001 --reward_type=00010 --deta_w=1. --deta_l=10. --agent_type=multi_agent --mcf_path=../inputs/MCFsolution/MCF_NSF_30_90.txt --stamp_type=ftest_ln_maxsessutil_maxutil_OBL_5_30_90

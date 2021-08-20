#!/bin/bash

# * BC phase of training the IL agent `L_K+L_V+L_F(c)` on the Leaderboard benchmark.
# train_il () {
# python -u train_il.py reset_step=true \
# agent.cilrs.wb_run_path=null agent.cilrs.wb_ckpt_step=null \
# wb_project="il_leaderboard_roach" wb_group="train0" 'wb_name="L_K+L_V+L_F(c)"' \
# dagger_datasets=["zhejun/il_leaderboard_roach/1pok3pwb"] \
# 'agent.cilrs.env_wrapper.kwargs.input_states=[speed,vec,cmd]' \
# agent.cilrs.policy.kwargs.number_of_branches=1 \
# agent.cilrs.training.kwargs.branch_weights=[1.0] \
# agent.cilrs.env_wrapper.kwargs.action_distribution="beta_shared" \
# agent.cilrs.rl_run_path=iccv21-roach/trained-models/1929isj0 agent.cilrs.rl_ckpt_step=11833344 \
# agent.cilrs.training.kwargs.action_kl=true \
# agent.cilrs.env_wrapper.kwargs.value_as_supervision=true \
# agent.cilrs.training.kwargs.value_weight=0.001 \
# agent.cilrs.env_wrapper.kwargs.dim_features_supervision=256 \
# agent.cilrs.training.kwargs.features_weight=0.05 \
# agent.cilrs.training.kwargs.batch_size=64 \
# cache_dir=${CACHE_DIR}
# }

# * BC phase of training the IL agent `L_A(AP)` on the NoCrash benchmark.
train_il () {
python -u train_il.py reset_step=true \
agent.cilrs.wb_run_path=null agent.cilrs.wb_ckpt_step=null \
wb_project="il_nocrash_ap" wb_group="train0" 'wb_name="L_A(AP)"' \
dagger_datasets=["zhejun/il_nocrash_ap/2pilkrol"] \
'agent.cilrs.env_wrapper.kwargs.input_states=[speed]' \
agent.cilrs.policy.kwargs.number_of_branches=6 \
agent.cilrs.training.kwargs.branch_weights=[1.0,1.0,1.0,1.0,1.0,1.0] \
agent.cilrs.env_wrapper.kwargs.action_distribution=null \
agent.cilrs.rl_run_path=null agent.cilrs.rl_ckpt_step=null \
agent.cilrs.training.kwargs.action_kl=false \
agent.cilrs.env_wrapper.kwargs.value_as_supervision=false \
agent.cilrs.training.kwargs.value_weight=0.0 \
agent.cilrs.env_wrapper.kwargs.dim_features_supervision=0 \
agent.cilrs.training.kwargs.features_weight=0.0 \
agent.cilrs.training.kwargs.batch_size=64 \
cache_dir=${CACHE_DIR}
}

# * DAGGER iteration 1 of training the IL agent `L_A(AP)` on the NoCrash benchmark.
# * All settings will be loaded from the checkpoint `zhejun/il_nocrash_ap/AGENT_TRAIN0`
# * Dataset `zhejun/il_nocrash_ap/DATA_DAGGER0` should be collected before this round of training.
# * It should be collected using the trained agent at training iteration 0 (BC).
# train_il () {
# python -u train_il.py reset_step=true \
# agent.cilrs.wb_run_path=zhejun/il_nocrash_ap/AGENT_TRAIN0 agent.cilrs.wb_ckpt_step=24 \
# wb_project="il_nocrash_ap" wb_group="train1" wb_name='"L_A(AP)"' \
# dagger_datasets=["zhejun/il_nocrash_ap/DATA_DAGGER0","zhejun/il_nocrash_ap/BC_DATA"] \
# cache_dir=${CACHE_DIR}
# }

# NO NEED TO MODIFY THE FOLLOWING
# actiate conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate carla

NODE_ROOT=/home/ubuntu/tmp_data
mkdir -p "${NODE_ROOT}"
CACHE_DIR=$(mktemp -d --tmpdir="${NODE_ROOT}")

echo "CACHE_DIR: ${CACHE_DIR}"

train_il

echo "Python finished!!"
rm -rf "${CACHE_DIR}"
echo "Bash script done!!"
echo finished at: `date`
exit 0;

# To shut down the aws instance after the script is finished
# sleep 10
# sudo shutdown -h now
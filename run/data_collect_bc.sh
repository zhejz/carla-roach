#!/bin/bash

# * To collect from Roach for the Leaderboard benchmark
data_collect () {
  python -u data_collect.py resume=true log_video=false save_to_wandb=true \
  wb_project=il_leaderboard_roach \
  wb_group=bc_data \
  test_suites=lb_data \
  n_episodes=160 \
  dataset_root=/home/ubuntu/dataset/bc \
  actors.hero.driver=ppo \
  agent.ppo.wb_run_path=iccv21-roach/trained-models/1929isj0 \
  agent.ppo.wb_ckpt_step=null \
  agent/cilrs/obs_configs=central_rgb_wide \
  inject_noise=true \
  actors.hero.terminal.kwargs.max_time=300 \
  actors.hero.terminal.kwargs.no_collision=true \
  actors.hero.terminal.kwargs.no_run_rl=false \
  actors.hero.terminal.kwargs.no_run_stop=false \
  carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh
}

# * To collect from Autopilot for the NoCrash benchmark
# data_collect () {
#   python -u data_collect.py resume=true log_video=false save_to_wandb=true \
#   wb_project=il_nocrash_ap \
#   wb_group=bc_data \
#   test_suites=eu_data \
#   n_episodes=80 \
#   dataset_root=/home/ubuntu/dataset/bc \
#   actors.hero.driver=roaming \
#   agent/cilrs/obs_configs=central_rgb_wide \
#   inject_noise=true \
#   actors.hero.terminal.kwargs.max_time=300 \
#   actors.hero.terminal.kwargs.no_collision=true \
#   actors.hero.terminal.kwargs.no_run_rl=false \
#   actors.hero.terminal.kwargs.no_run_stop=false \
#   carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh
# }


# NO NEED TO MODIFY THE FOLLOWING
# actiate conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate carla

# remove checkpoint files
rm outputs/checkpoint.txt
rm outputs/wb_run_id.txt
rm outputs/ep_stat_buffer_*.json

# resume benchmark in case carla is crashed.
RED=$'\e[0;31m'
NC=$'\e[0m'
PYTHON_RETURN=1
until [ $PYTHON_RETURN == 0 ]; do
  data_collect
  PYTHON_RETURN=$?
  echo "${RED} PYTHON_RETURN=${PYTHON_RETURN}!!! Start Over!!!${NC}" >&2
  sleep 2
done

killall -9 -r CarlaUE4-Linux
echo "Bash script done."

# To shut down the aws instance after the script is finished
# sleep 10
# sudo shutdown -h now
#!/bin/bash

# * To benchmark the Autopilot.
agent="roaming"
benchmark () {
  python -u benchmark.py resume=true log_video=true \
  wb_project=iccv21-roach-benchmark \
  agent=$agent actors.hero.agent=$agent \
  +agent/roaming/obs_configs=birdview \
  'wb_group="Autopilot"' \
  'wb_notes="Benchmark Autopilot on NoCrash-dense."' \
  test_suites=nocrash_dense \
  seed=2021 \
  +wb_sub_group=nocrash_dense-2021 \
  no_rendering=true \
  carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh
}

# * To benchmark rl experts.
# agent="ppo"
# benchmark () {
#   python -u benchmark.py resume=true log_video=true \
#   wb_project=iccv21-roach-benchmark \
#   agent=$agent actors.hero.agent=$agent \
#   agent.ppo.wb_run_path=iccv21-roach/trained-models/1929isj0 \
#   'wb_group="Roach"' \
#   'wb_notes="Benchmark Roach on NoCrash-dense."' \
#   test_suites=nocrash_dense \
#   seed=2021 \
#   +wb_sub_group=nocrash_dense-2021 \
#   no_rendering=true \
#   carla_sh_path=${CARLA_ROOT}/CarlaUE4.sh
# }

# * To benchmark il agents.
# agent="cilrs"
# benchmark () {
#   python -u benchmark.py resume=true log_video=true \
#   wb_project=iccv21-roach-benchmark \
#   agent=$agent actors.hero.agent=$agent \
#   agent.cilrs.wb_run_path=iccv21-roach/trained-models/31u9tki7 \
#   'wb_group="L_K+L_F(c)"' \
#   'wb_notes="Benchmark L_K+L_F(c) on NoCrash-dense."' \
#   test_suites=nocrash_dense \
#   seed=2021 \
#   +wb_sub_group=nocrash_dense-2021 \
#   no_rendering=false \
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
  benchmark
  PYTHON_RETURN=$?
  echo "${RED} PYTHON_RETURN=${PYTHON_RETURN}!!! Start Over!!!${NC}" >&2
  sleep 2
done

killall -9 -r CarlaUE4-Linux
echo "Bash script done."

# To shut down the aws instance after the script is finished
# sleep 10
# sudo shutdown -h now
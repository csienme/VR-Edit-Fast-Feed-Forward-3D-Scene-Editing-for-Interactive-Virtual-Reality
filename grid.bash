# temp.bash 本身不加 nohup，維持你現在的寫法
#!/bin/bash
set -e

function="1" bash run_spinnerf.sh
function="2" bash run_spinnerf.sh
function="3" bash run_spinnerf.sh

python grid_search.py \
    --base_config configs/exp_baseline.yaml \
    --mode        train_only \
    --n_trials    40 \
    --study_name  phase1_train
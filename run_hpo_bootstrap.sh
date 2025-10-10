#!/bin/bash

# CatBoost HPO with different bootstrap types
# Each bootstrap type will run 300 trials

echo "======================================================================"
echo "🚀 Starting CatBoost HPO for all bootstrap types"
echo "======================================================================"
echo ""

# Array of bootstrap types
BOOTSTRAP_TYPES=("Bernoulli" "Bayesian" "MVS")

# HPO parameters
N_TRIALS=100
TRAIN_T_PATH="data/proc_train_t"
TRAIN_V_PATH="data/proc_train_v"
TRAIN_C_PATH="data/proc_train_c"
OUTPUT_CONFIG="config_optimized.yaml"
ORIGINAL_CONFIG="config_GBDT.yaml"

# Loop through each bootstrap type
for BOOTSTRAP_TYPE in "${BOOTSTRAP_TYPES[@]}"; do
    echo ""
    echo "======================================================================"
    echo "🔥 Running HPO for Bootstrap Type: ${BOOTSTRAP_TYPE}"
    echo "======================================================================"
    echo ""
    
    # Run HPO
    CUDA_VISIBLE_DEVICES=1 python hpo_catboost.py \
        --bootstrap-types ${BOOTSTRAP_TYPE} \
        --n-trials ${N_TRIALS} \
        --train-t-path ${TRAIN_T_PATH} \
        --train-v-path ${TRAIN_V_PATH} \
        --train-c-path ${TRAIN_C_PATH} \
        --task-type GPU \
        --output-config ${OUTPUT_CONFIG} \
        --original-config ${ORIGINAL_CONFIG}
    
    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Successfully completed HPO for ${BOOTSTRAP_TYPE}"
        echo ""
    else
        echo ""
        echo "❌ Failed HPO for ${BOOTSTRAP_TYPE}"
        echo ""
        exit 1
    fi
    
    # Wait a bit between runs to let GPU cool down
    echo "⏳ Waiting 10 seconds before next run..."
    sleep 10
done

echo ""
echo "======================================================================"
echo "🎉 All HPO runs completed successfully!"
echo "======================================================================"
echo ""
echo "Generated config files:"
for BOOTSTRAP_TYPE in "${BOOTSTRAP_TYPES[@]}"; do
    echo "  - config_optimized_${BOOTSTRAP_TYPE}_catboost_best.yaml"
done
echo ""


#!/bin/bash

#SBATCH --job-name=cbtm_inference
#SBATCH --output=/home/ehghaghi/scratch/ehghaghi/inference/run_logs/%x_%j.out
#SBATCH --error=/home/ehghaghi/scratch/ehghaghi/inference/run_logs/%x_%j.err
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH --account=aip-craffel
#SBATCH --time=23:00:00

# srun -c 4 --gres=gpu:2 --partition l40 --mem=128GB --pty --time=16:00:00 bash

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU resources: $CUDA_VISIBLE_DEVICES"

# Load modules and activate environment
module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11
source /home/ehghaghi/projects/aip-craffel/ehghaghi/c-btm-distillation/uv-x86_64-unknown-linux-gnu/distill_env/bin/activate


# Export cache dirs
export SCRATCH="/home/ehghaghi/scratch/ehghaghi"
export HUGGINGFACE_HUB_CACHE=$SCRATCH/.cache

set -e  # Exit on any error

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

# Model paths - update these to match your saved files
VECTORIZER_PATH="$SCRATCH/clusters/allenai/tulu-3-sft-mixture/8/tfidf.pkl"
CLUSTER_CENTERS_PATH="$SCRATCH/clusters/allenai/tulu-3-sft-mixture/8/kmeans.pkl"
EXPERT_MODELS_DIR="$SCRATCH/distillation_results"

# Script path
CBTM_SCRIPT="ensemble.py"

# Default parameters
DEFAULT_TEMPERATURE="0.1"
DEFAULT_TOP_K="2"
DEFAULT_MAX_TOKENS="512"


# =============================================================================
# YOUR CONTEXTS - MODIFY THESE FOR YOUR EXPERIMENTS
# =============================================================================

# Main context for single inference
MAIN_CONTEXT="Solve this differential equation: dy/dx = x^2 + 1"

# Context for temperature analysis
TEMP_ANALYSIS_CONTEXT="Implement a machine learning model using PyTorch"

# Context for top-k analysis  
TOPK_ANALYSIS_CONTEXT="Explain how backpropagation works in neural networks"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

print_banner() {
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
}

print_section() {
    echo ""
    echo "--------------------------------------------------------------------------------"
    echo "$1"
    echo "--------------------------------------------------------------------------------"
}

check_requirements() {
    print_section "🔍 Checking C-BTM Requirements"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo "❌ Python3 not found"
        exit 1
    fi
    
    # Check script
    if [ ! -f "$CBTM_SCRIPT" ]; then
        echo "❌ C-BTM script not found: $CBTM_SCRIPT"
        exit 1
    fi
    
    # Check vectorizer
    if [ ! -f "$VECTORIZER_PATH" ]; then
        echo "❌ Vectorizer not found: $VECTORIZER_PATH"
        exit 1
    fi
    
    # Check cluster centers
    if [ ! -f "$CLUSTER_CENTERS_PATH" ]; then
        echo "❌ Cluster centers not found: $CLUSTER_CENTERS_PATH"
        exit 1
    fi
    
    # Check expert models directory
    if [ ! -d "$EXPERT_MODELS_DIR" ]; then
        echo "❌ Expert models directory not found: $EXPERT_MODELS_DIR"
        exit 1
    fi
    
    # Count and verify expert checkpoint directories
    expert_dirs=($(find "$EXPERT_MODELS_DIR" -maxdepth 1 -type d -name "*" | grep -v "^$EXPERT_MODELS_DIR$" | sort))
    expert_count=${#expert_dirs[@]}
    
    if [ "$expert_count" -eq 0 ]; then
        echo "❌ No expert checkpoint directories found in $EXPERT_MODELS_DIR"
        exit 1
    fi
    
    echo "✅ Found $expert_count expert checkpoint directories:"
    for i in "${!expert_dirs[@]}"; do
        dir_name=$(basename "${expert_dirs[$i]}")
        echo "   [$i] $dir_name"
        
        # Check if directory contains model files
        if [ ! -f "${expert_dirs[$i]}/pytorch_model.bin" ] && [ ! -f "${expert_dirs[$i]}/model.safetensors" ]; then
            echo "      ⚠️  No model files found (pytorch_model.bin or model.safetensors)"
        fi
    done
    
    # Verify cluster-expert correspondence
    echo ""
    echo "🔍 Verifying cluster-expert correspondence..."
    echo "Expected: expert directories should correspond to cluster indices 0 to $((expert_count-1))"
    echo "Actual order (after sorting):"
    for i in "${!expert_dirs[@]}"; do
        echo "   Cluster $i → $(basename "${expert_dirs[$i]}")"
    done
    
    echo ""
    echo "⚠️  IMPORTANT: Ensure the sorted order above matches your cluster training!"
    echo "   Expert at index $i should be trained on cluster $i data"
    
    echo "✅ All requirements satisfied"
}

# =============================================================================
# C-BTM INFERENCE FUNCTIONS
# =============================================================================

run_cbtm_inference() {
    local context="$1"
    local temperature="${2:-$DEFAULT_TEMPERATURE}"
    local top_k="${3:-$DEFAULT_TOP_K}"
    local max_tokens="${4:-$DEFAULT_MAX_TOKENS}"
    
    print_section "🧠 C-BTM Sparse Ensemble Inference"
    echo "Context: \"$context\""
    echo "Temperature: $temperature"
    echo "Top-K Experts: $top_k"
    echo "Max Tokens: $max_tokens"
    echo ""
    
    python3 "$CBTM_SCRIPT" \
        --vectorizer "$VECTORIZER_PATH" \
        --cluster-centers "$CLUSTER_CENTERS_PATH" \
        --expert-dir "$EXPERT_MODELS_DIR" \
        --context "$context" \
        --temperature "$temperature" \
        --top-k "$top_k" \
        --max-tokens "$max_tokens"
}

run_ensemble_weights_only() {
    local context="$1"
    local temperature="${2:-$DEFAULT_TEMPERATURE}"
    local top_k="${3:-$DEFAULT_TOP_K}"
    
    print_section "⚖️ Computing Ensemble Weights Only"
    echo "Context: \"$context\""
    echo ""
    
    python3 "$CBTM_SCRIPT" \
        --vectorizer "$VECTORIZER_PATH" \
        --cluster-centers "$CLUSTER_CENTERS_PATH" \
        --expert-dir "$EXPERT_MODELS_DIR" \
        --context "$context" \
        --temperature "$temperature" \
        --top-k "$top_k" \
        --max-tokens 0  # Don't generate, just show weights
}

run_temperature_sweep() {
    local context="$1"
    
    print_section "🌡️ Temperature Sweep Analysis"
    echo "Context: \"$context\""
    echo ""
    
    declare -a temps=("0.1" "0.2" "0.5" "1.0")
    
    for temp in "${temps[@]}"; do
        echo "Temperature: $temp"
        echo "----------------------------------------"
        
        python3 "$CBTM_SCRIPT" \
            --vectorizer "$VECTORIZER_PATH" \
            --cluster-centers "$CLUSTER_CENTERS_PATH" \
            --expert-dir "$EXPERT_MODELS_DIR" \
            --context "$context" \
            --temperature "$temp" \
            --top-k "$DEFAULT_TOP_K" \
            --max-tokens 0
        echo ""
    done
}

run_top_k_sweep() {
    local context="$1"
    local temperature="${2:-$DEFAULT_TEMPERATURE}"
    
    print_section "🔝 Top-K Sweep Analysis"
    echo "Context: \"$context\""
    echo "Temperature: $temperature"
    echo ""
    
    # Get number of available experts
    expert_count=$(find "$EXPERT_MODELS_DIR" -name "*.pkl" | wc -l)
    
    declare -a top_ks=("1" "2" "4")
    if [ "$expert_count" -gt 4 ]; then
        top_ks+=("$expert_count")  # Add "all experts" option
    fi
    
    for k in "${top_ks[@]}"; do
        echo "Top-K: $k"
        echo "----------------------------------------"
        
        python3 "$CBTM_SCRIPT" \
            --vectorizer "$VECTORIZER_PATH" \
            --cluster-centers "$CLUSTER_CENTERS_PATH" \
            --expert-dir "$EXPERT_MODELS_DIR" \
            --context "$context" \
            --temperature "$temperature" \
            --top-k "$k" \
            --max-tokens 0
        echo ""
    done
}

run_batch_examples() {
    print_section "📦 Batch Example Contexts"
    
    # Array of test contexts - MODIFY THESE FOR YOUR EXPERIMENTS
    declare -a test_contexts=(
        "Solve this differential equation: dy/dx = x^2 + 1"
        "Implement a binary search algorithm in Python"
        "Explain the concept of gradient descent optimization"
        "Write a function to calculate fibonacci numbers recursively"
        "How do transformers work in natural language processing?"
        "Debug this code: for i in range(10) print(i)"
        "What is the time complexity of quicksort algorithm?"
        "Explain quantum entanglement in simple terms"
        "Create a REST API endpoint using FastAPI"
        "How do I optimize neural network training?"
    )
    
    for i in "${!test_contexts[@]}"; do
        context="${test_contexts[$i]}"
        echo "[$((i+1))/${#test_contexts[@]}] Context: \"$context\""
        echo "----------------------------------------"
        
        python3 "$CBTM_SCRIPT" \
            --vectorizer "$VECTORIZER_PATH" \
            --cluster-centers "$CLUSTER_CENTERS_PATH" \
            --expert-dir "$EXPERT_MODELS_DIR" \
            --context "$context" \
            --temperature "$DEFAULT_TEMPERATURE" \
            --top-k "$DEFAULT_TOP_K" \
            --max-tokens 0
        echo ""
    done
}

interactive_mode() {
    print_section "💬 Interactive C-BTM Mode"
    echo "Enter contexts for C-BTM inference (type 'quit' to exit)"
    echo "Commands:"
    echo "  - Just text: runs inference with default settings"
    echo "  - 'temp X.X <text>': sets temperature X.X"
    echo "  - 'topk N <text>': sets top-k to N"
    echo "  - 'weights <text>': shows only ensemble weights"
    echo ""
    
    while true; do
        read -p "📝 Enter context (or command): " user_input
        
        if [ "$user_input" = "quit" ] || [ "$user_input" = "exit" ]; then
            echo "👋 Goodbye!"
            break
        fi
        
        if [ -z "$user_input" ]; then
            echo "⚠️ Please enter some text"
            continue
        fi
        
        # Parse commands
        if [[ $user_input == temp\ * ]]; then
            temp=$(echo "$user_input" | cut -d' ' -f2)
            context=$(echo "$user_input" | cut -d' ' -f3-)
            run_cbtm_inference "$context" "$temp"
        elif [[ $user_input == topk\ * ]]; then
            topk=$(echo "$user_input" | cut -d' ' -f2)
            context=$(echo "$user_input" | cut -d' ' -f3-)
            run_cbtm_inference "$context" "$DEFAULT_TEMPERATURE" "$topk"
        elif [[ $user_input == weights\ * ]]; then
            context=$(echo "$user_input" | cut -d' ' -f2-)
            run_ensemble_weights_only "$context"
        else
            run_cbtm_inference "$user_input"
        fi
        echo ""
    done
}

# =============================================================================
# MAIN MENU
# =============================================================================

show_menu() {
    echo ""
    echo "🧠 C-BTM INFERENCE MENU"
    echo "======================="
    echo "1. Check requirements"
    echo "2. Single context inference"
    echo "3. Show ensemble weights only"
    echo "4. Temperature sweep"
    echo "5. Top-K sweep"
    echo "6. Batch examples"
    echo "7. Interactive mode"
    echo "8. Custom inference"
    echo "0. Exit"
    echo ""
}

main_menu() {
    while true; do
        show_menu
        read -p "Select option (0-8): " choice
        
        case $choice in
            1)
                check_requirements
                ;;
            2)
                read -p "Enter context: " context
                run_cbtm_inference "$context"
                ;;
            3)
                read -p "Enter context: " context
                run_ensemble_weights_only "$context"
                ;;
            4)
                read -p "Enter context: " context
                run_temperature_sweep "$context"
                ;;
            5)
                read -p "Enter context: " context
                read -p "Temperature [$DEFAULT_TEMPERATURE]: " temp
                temp=${temp:-$DEFAULT_TEMPERATURE}
                run_top_k_sweep "$context" "$temp"
                ;;
            6)
                run_batch_examples
                ;;
            7)
                interactive_mode
                ;;
            8)
                echo "Custom C-BTM inference:"
                read -p "Context: " custom_context
                read -p "Temperature [$DEFAULT_TEMPERATURE]: " custom_temp
                read -p "Top-K [$DEFAULT_TOP_K]: " custom_k
                read -p "Max tokens [$DEFAULT_MAX_TOKENS]: " custom_tokens
                
                custom_temp=${custom_temp:-$DEFAULT_TEMPERATURE}
                custom_k=${custom_k:-$DEFAULT_TOP_K}
                custom_tokens=${custom_tokens:-$DEFAULT_MAX_TOKENS}
                
                run_cbtm_inference "$custom_context" "$custom_temp" "$custom_k" "$custom_tokens"
                ;;
            0)
                echo "👋 Goodbye!"
                exit 0
                ;;
            *)
                echo "❌ Invalid option. Please select 0-8."
                ;;
        esac
        
        echo ""
        read -p "Press Enter to continue..."
    done
}

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

print_banner "🧠 C-BTM SPARSE ENSEMBLE INFERENCE"
echo "SLURM Job: $SLURM_JOB_ID on $(hostname)"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"
echo ""

# Parse command line arguments
if [ $# -eq 0 ]; then
    # No arguments - run default analysis automatically
    echo "🚀 Running automatic C-BTM analysis (no interactive menu on SLURM)"
    echo ""
    
    # Check requirements first
    check_requirements
    
    # # Run batch examples by default
    # echo "📦 Running batch examples..."
    # run_batch_examples
    
    # Show temperature analysis for predefined context
    echo ""
    echo "🌡️ Running temperature analysis..."
    run_temperature_sweep "$TEMP_ANALYSIS_CONTEXT"
    
    # Show top-k analysis for predefined context
    echo ""
    echo "🔝 Running top-k analysis..."
    run_top_k_sweep "$TOPK_ANALYSIS_CONTEXT" "0.1"
    
    echo ""
    echo "✅ C-BTM analysis complete!"
    
else
    case "$1" in
        "check")
            check_requirements
            ;;
        "infer")
            check_requirements
            if [ -z "$2" ]; then
                echo "❌ Usage: $0 infer \"context text\""
                exit 1
            fi
            run_cbtm_inference "$2" "${3:-$DEFAULT_TEMPERATURE}" "${4:-$DEFAULT_TOP_K}" "${5:-$DEFAULT_MAX_TOKENS}"
            ;;
        "weights")
            check_requirements
            if [ -z "$2" ]; then
                echo "❌ Usage: $0 weights \"context text\""
                exit 1
            fi
            run_ensemble_weights_only "$2" "${3:-$DEFAULT_TEMPERATURE}" "${4:-$DEFAULT_TOP_K}"
            ;;
        "temp-sweep")
            check_requirements
            if [ -z "$2" ]; then
                echo "❌ Usage: $0 temp-sweep \"context text\""
                exit 1
            fi
            run_temperature_sweep "$2"
            ;;
        "topk-sweep")
            check_requirements
            if [ -z "$2" ]; then
                echo "❌ Usage: $0 topk-sweep \"context text\""
                exit 1
            fi
            run_top_k_sweep "$2" "${3:-$DEFAULT_TEMPERATURE}"
            ;;
        "batch")
            check_requirements
            run_batch_examples
            ;;
        "demo")
            check_requirements
            echo "🚀 Running C-BTM demo analysis..."
            echo "Main context: $MAIN_CONTEXT"
            echo ""
            run_cbtm_inference "$MAIN_CONTEXT"
            run_batch_examples
            run_temperature_sweep "$TEMP_ANALYSIS_CONTEXT"
            run_top_k_sweep "$TOPK_ANALYSIS_CONTEXT" "0.1"
            ;;
        *)
            echo "❌ Unknown command: $1"
            echo ""
            echo "Available commands:"
            echo "  check                    - Check requirements"
            echo "  infer \"text\" [temp] [k] [tokens] - Run inference"
            echo "  weights \"text\" [temp] [k]        - Show ensemble weights"
            echo "  temp-sweep \"text\"              - Temperature analysis"
            echo "  topk-sweep \"text\" [temp]       - Top-K analysis"
            echo "  batch                    - Run batch examples"
            echo "  demo                     - Run comprehensive demo"
            echo ""
            echo "Examples:"
            echo "  sbatch $0                                    # Auto analysis"
            echo "  sbatch $0 infer \"Solve this equation\"       # Single inference"
            echo "  sbatch $0 batch                             # Batch examples"
            echo "  sbatch $0 demo                              # Full demo"
            exit 1
            ;;
    esac
fi
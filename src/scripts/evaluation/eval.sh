#!/bin/bash

#SBATCH --job-name=eval_distilled_model
#SBATCH --output=/home/ehghaghi/scratch/ehghaghi/distil_evals/run_logs/%x_%j.out
#SBATCH --error=/home/ehghaghi/scratch/ehghaghi/distil_evals/run_logs/%x_%j.err
#SBATCH --partition=gpubase_l40s_b3
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --account=aip-craffel
#SBATCH --time=12:00:00

# Usage: sbatch eval_slurm.sh [model_path]
# Example: sbatch eval_slurm.sh /home/ehghaghi/scratch/ehghaghi/distillation_results/0

echo "Job ${SLURM_JOB_NAME} (${SLURM_JOB_ID}) started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU resources: $CUDA_VISIBLE_DEVICES"
echo "Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits

# Load modules and activate environment
echo "Loading modules and environment..."
module load cuda/12.6
module load gcc arrow/19.0.1 python/3.11
source /home/ehghaghi/projects/aip-craffel/ehghaghi/c-btm-distillation/uv-x86_64-unknown-linux-gnu/distill_env/bin/activate

# Export cache directories
export SCRATCH="/home/ehghaghi/scratch/ehghaghi"
export HF_HOME=$SCRATCH/.cache

# Default model path (can be overridden by command line argument)
# DEFAULT_MODEL_PATH="$SCRATCH/distillation_results/7"
DEFAULT_MODEL_PATH="$SCRATCH/Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DATA_SPLIT="programming_and_code_development"

# Get model path from command line argument or use default
MODEL_PATH="${1:-$DEFAULT_MODEL_PATH}"
DATA_PATH="${2:-$DEFAULT_DATA_SPLIT}"

echo ""
echo "🔍 Evaluating Distilled Student Model"
echo "====================================="
echo "📂 Model Path: $MODEL_PATH"
echo "📂 Data Path: $DATA_PATH"
echo "🕐 Start Time: $(date)"
echo "💾 Memory: ${SLURM_MEM_PER_NODE}MB"
echo "🖥️  CPUs: ${SLURM_CPUS_PER_TASK}"
echo ""

# Verify model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ ERROR: Model path does not exist: $MODEL_PATH"
    echo "Available distillation results:"
    ls -la $SCRATCH/distillation_results/ || echo "No distillation results directory found"
    exit 1
fi

# Check if model files exist
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "❌ ERROR: config.json not found in $MODEL_PATH"
    echo "Contents of model directory:"
    ls -la "$MODEL_PATH"
    exit 1
fi

echo "✅ Model validation passed"
echo "📁 Model directory contents:"
ls -la "$MODEL_PATH"
echo ""

# Set Python environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export TOKENIZERS_PARALLELISM=false  # Avoid tokenizer warnings in multiprocessing

# Create evaluation logs directory
EVAL_LOGS_DIR="$SCRATCH/eval_logs"
mkdir -p "$EVAL_LOGS_DIR"

# Generate timestamped log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EVAL_LOG_FILE="$EVAL_LOGS_DIR/eval_${SLURM_JOB_ID}_${TIMESTAMP}.log"

echo "📝 Detailed logs will be saved to: $EVAL_LOG_FILE"
echo ""

# Print Python environment info
echo "🐍 Python Environment Info:"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Transformers version: $(python -c 'import transformers; print(transformers.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Run the evaluation with detailed logging
echo "🚀 Starting Cross-Entropy Evaluation..."
echo "======================================="

# Run evaluation with both stdout/stderr capture and file logging
python eval_lm_loss.py "$MODEL_PATH" "$DATA_PATH" 2>&1 | tee "$EVAL_LOG_FILE"

# Check if evaluation completed successfully
EVAL_EXIT_CODE=${PIPESTATUS[0]}

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Evaluation Completed Successfully!"
    echo "===================================="
    
    # Show results summary
    RESULTS_FILE="$MODEL_PATH/eval_results.json"
    if [ -f "$RESULTS_FILE" ]; then
        echo "📊 Results Summary:"
        echo "-------------------"
        python -c "
import json
try:
    with open('$RESULTS_FILE', 'r') as f:
        results = json.load(f)
    print(f'📈 Test Dataset Size: {results[\"test_dataset_size\"]:,} examples')
    print(f'📉 Average Cross-Entropy Loss: {results[\"average_cross_entropy_loss\"]:.6f}')
    print(f'📊 Perplexity: {results[\"perplexity\"]:.4f}')
    print(f'⚙️  Number of Batches: {results[\"num_batches\"]:,}')
except Exception as e:
    print(f'❌ Error reading results: {e}')
"
        echo ""
        echo "📁 Full results saved to: $RESULTS_FILE"
    else
        echo "⚠️  Warning: Results file not found at $RESULTS_FILE"
    fi
    
    echo "📝 Detailed log: $EVAL_LOG_FILE"
    echo "🕐 End Time: $(date)"
    
else
    echo ""
    echo "❌ Evaluation Failed!"
    echo "==================="
    echo "Exit code: $EVAL_EXIT_CODE"
    echo "📝 Check detailed log for errors: $EVAL_LOG_FILE"
    echo "🕐 Failed at: $(date)"
    exit $EVAL_EXIT_CODE
fi

echo ""
echo "🎉 Job completed successfully!"
echo "Job ID: $SLURM_JOB_ID"
echo "Total runtime: $SECONDS seconds"
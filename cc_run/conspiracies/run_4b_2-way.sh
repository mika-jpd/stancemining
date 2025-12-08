#!/bin/bash
#SBATCH --job-name=4b-2way
#SBATCH --mem=280G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=48:0:0
#SBATCH --gres=gpu:h100:1
#SBATCH --mail-user=mika.desblancs@mcgill.ca
#SBATCH --mail-type=ALL

cd "$project/stancemining_forked_mika/"

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.6
module load gcc opencv arrow rust

virtualenv --clear --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

echo "=== DEBUG MODULE ENVIRONMENT ==="
echo "Node: $(hostname)"
echo "Which python: $(which python)"
echo "Which python3: $(which python3)"
echo "Python version: $(python --version 2>&1)"
echo "Python3 version: $(python3 --version 2>&1)"
echo "PATH: $PATH"
echo "PYTHONPATH: $PYTHONPATH"
module list
echo "================================"

#unset CUDA_VISIBLE_DEVICES # need to test if this is required

pip install python-dotenv transformers wandb hydra-core sentence-transformers accelerate datasets evaluate peft bert-score sacrebleu nltk vllm gpytorch pyro-api pyro-ppl
pip install --no-index polars

python -m experiments.scripts.train_model finetune.task=claim-entailment-2way data.dataset=[stanceosaurus,conspiracies] finetune.model_name=Qwen/Qwen3-4B finetune.batch_size=32 finetune.grad_accum_steps=2 finetune.classification_method=head finetune.num_epochs=3 finetune.lora_r=64 finetune.lora_alpha=128 finetune.attn_implementation=null

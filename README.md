# LoHRbench: OpenVLA Baseline

Training and evaluation code for the **OpenVLA** (7B) baseline on LoHRbench, built on top of [OpenVLA-OFT](https://github.com/openvla/openvla-oft).

GitHub: [https://github.com/HaoranZhangumich/LOHRbench_Openvla](https://github.com/HaoranZhangumich/LOHRbench_Openvla)

## Repository Structure

```
LOHRbench_Openvla/
├── openvla-oft/                        # OpenVLA-OFT framework (modified)
│   ├── vla-scripts/
│   │   ├── finetune.py                 # LoRA fine-tuning script
│   │   ├── deploy.py                   # Model deployment
│   │   └── merge_lora_weights_and_save.py  # Merge LoRA adapters
│   ├── prismatic/
│   │   ├── models/
│   │   │   ├── vlas/openvla.py         # OpenVLA model
│   │   │   ├── action_heads.py         # L1 regression / diffusion heads
│   │   │   └── projectors.py           # Proprioceptive projector
│   │   ├── vla/
│   │   │   ├── datasets/               # RLDS data loading
│   │   │   └── constants.py            # Action/state dimensions
│   │   └── conf/
│   │       └── vla.py                  # VLA experiment configs
│   └── experiments/robot/
│       └── openvla_utils.py            # Model loading utilities
├── evaluate_openvla_debug.py                 # openvla offline evaluation script
└── README.md
```

## Setup

There are two setup paths depending on your use case.

### Path 1: Standalone OpenVLA env (training + offline eval only)

Use this if you only need to train OpenVLA or run standalone offline evaluation (`evaluate_openvla_debug.py`). This does **not** require lohrbench or ManiSkill.

```bash
cd openvla-oft

# Create conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# Install PyTorch (CUDA 12.4)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install Flash Attention 2
pip install packaging ninja
pip install flash-attn==2.5.5 --no-build-isolation

# Install OpenVLA
pip install -e .
```

See [`openvla-oft/SETUP.md`](openvla-oft/SETUP.md) for detailed setup instructions.

### Path 2: Combined env with lohrbench (for unified simulation-based evaluation)

Use this if you need to run `baseline/eval.py --policy openvla`, which requires both lohrbench and openvla in the same Python environment.

**Prerequisite:** You must have a working lohrbench conda environment set up per the [main repo README](../README.md) (with ManiSkill, SAPIEN, mplib, pddl, etc.).

```bash
# Activate your existing lohrbench conda environment
conda activate <your_lohrbench_env>

# Install openvla-oft WITHOUT dependencies (avoids torch version conflict)
cd LOHRbench_Openvla/openvla-oft
pip install --no-deps -e .

# Install openvla-specific dependencies not already in lohrbench
pip install accelerate "draccus==0.8.0" einops huggingface_hub json-numpy jsonlines \
    "peft==0.11.1" protobuf rich "sentencepiece==0.1.99" "timm==0.9.10" \
    "tokenizers==0.19.1" wandb "tensorflow==2.15.0" "tensorflow_datasets==4.9.3" \
    "tensorflow_graphics==2021.12.3" "diffusers==0.30.3" imageio uvicorn fastapi

# Install custom transformers fork (required — do NOT use stock transformers)
pip install "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git"

# Install custom dlimp fork
pip install "dlimp @ git+https://github.com/moojink/dlimp_openvla"

# Install Flash Attention 2
pip install packaging ninja
pip install flash-attn==2.5.5 --no-build-isolation

# Set OPENVLA_ROOT (add to your shell profile for persistence)
export OPENVLA_ROOT="/path/to/LOHRbench_Openvla/openvla-oft"
```

> **Note:** `pip install --no-deps -e .` registers the openvla-oft package without overwriting the existing torch version in the lohrbench env. The lohrbench env's torch is compatible with openvla-oft in practice despite the version difference.

## Dataset

Download the demonstration dataset from HuggingFace: **[oldTOM/LoHRbench](https://huggingface.co/datasets/oldTOM/LoHRbench)**

The HuggingFace dataset provides HDF5 files. OpenVLA requires **RLDS format**, so you need to convert the HDF5 data first (see below).

## Data Format

OpenVLA consumes data in **RLDS format** (TensorFlow Datasets). Convert the downloaded HDF5 trajectories to RLDS using the conversion script in [`baseline/utils/data_convert.py`](../baseline/utils/data_convert.py).

Expected RLDS data directory:
```
/data1/LoHRbench_rlds/
└── lohrbench_rlds/            # --data_root_dir points here
    └── lohrbench_rlds/        # dataset subdirectory (matches --dataset_name)
        └── 0.1.0/
            └── ...  (TFRecord files)
```

## Training

```bash
cd openvla-oft

python -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=1 \
    vla-scripts/finetune.py \
    --vla_path openvla/openvla-7b \
    --data_root_dir /data1/LoHRbench_rlds/lohrbench_rlds \
    --dataset_name lohrbench_rlds \
    --run_root_dir /data1/checkpoints/openvla \
    --use_lora True \
    --lora_rank 32 \
    --batch_size 8 \
    --grad_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --image_aug True \
    --num_images_in_input 2 \
    --use_proprio True \
    --save_freq 10000 \
    --wandb_project juicer_test \
    --wandb_entity your_entity \
    --run_id_note lohrbench_lora_r32
```

**Key hyperparameters:**

| Parameter | Value |
|---|---|
| Base model | `openvla/openvla-7b` |
| Action head | L1 regression |
| Input images | 2 (base + wrist camera) |
| Proprioceptive input | Enabled |
| LoRA rank | 32 |
| LoRA dropout | 0.0 |
| Batch size (per GPU) | 8 |
| Learning rate | 1e-4 |
| LR warmup steps | 0 |
| LR decay | 10x decay after 100k steps |
| Gradient accumulation | 1 |
| Image augmentation | Enabled |
| FiLM conditioning | Disabled |
| Shuffle buffer size | 100,000 |
| Save frequency | 10,000 steps |
| GPU | 1x NVIDIA A100 80GB |

Checkpoints are saved every 10,000 steps. The directory structure:
```
/data1/checkpoints/openvla/
└── openvla-7b+lohrbench_rlds+...+<step>_chkpt/
    ├── lora_adapter/                              # LoRA adapter directory
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    ├── action_head--<step>_checkpoint.pt           # Action head weights
    ├── proprio_projector--<step>_checkpoint.pt     # Proprioception projector weights
    └── dataset_statistics.json                    # Normalization statistics
```

### Merge LoRA Weights (Optional)

To merge LoRA adapters into the base model for faster inference:

```bash
python vla-scripts/merge_lora_weights_and_save.py \
    --base_checkpoint openvla/openvla-7b \
    --lora_finetuned_checkpoint_dir /path/to/checkpoint
```

## Evaluation

### Standalone evaluation (in this repo)

```bash
export OPENVLA_ROOT="/path/to/LOHRbench_Openvla/openvla-oft"

python evaluate_openvla.py \
    --checkpoint /path/to/checkpoint \
    --step 100000 \
    --benchmark-root /path/to/TAMPBench/benchmark/table-top \
    --task-types tool_using \
    --task-names repackage \
    --merge-lora
```

### Server-based evaluation

For multi-GPU or memory-constrained setups, use the server/client architecture:

```bash
# Terminal 1: Start the OpenVLA server
python openvla_server.py --checkpoint /path/to/checkpoint --step 100000

# Terminal 2: Run the evaluation client
python lohrbench_client.py --benchmark-root /path/to/TAMPBench/benchmark/table-top
```

### Unified evaluation (via LoHRBench)

```bash
export OPENVLA_ROOT="/path/to/LOHRbench_Openvla/openvla-oft"

python TAMPBench/baseline/eval.py \
    --policy openvla \
    --checkpoint /path/to/checkpoint \
    --step 100000 \
    --benchmark-root /path/to/TAMPBench/benchmark/table-top \
    --use-action-chunking --chunk-size 8 \
    --merge-lora \
    --results-dir ./results --save-video
```

See the [evaluation README](../TAMPBench/baseline/README.md) for full argument documentation.

## Acknowledgements

Built on top of [OpenVLA-OFT](https://github.com/openvla/openvla-oft).

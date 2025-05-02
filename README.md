# Robust-2D-DPO

Direct Preference Optimization (DPO) aligns large language models (LLMs) with human preferences but lacks the expressiveness of granular scoring. **2D-DPO** improves on this with a **two-dimensional segment-level scoring system**. However, 2D-DPO is sensitive to label noise, which can arise from imperfect human annotations or noisy automated labels.

This work explores:
- The advantages of 2D-DPO over DPO.
- The impact of noise on training stability and model alignment.
- A **robust version** of 2D-DPO designed to resist noisy signal degradation, supported by theoretical analysis and empirical results.
- Future directions for robust alignment methods with structured feedback.
  
  ---
- ## 1. Supervised Fine-Tuning (SFT)
- **Dataset**: `Noisy_2D-DPO.csv` (6,400 rows)
- **Format**: `_`, `instruction`, `chosen`, `rejected`, `chosen_segment_scores`, `rejected_segment_scores`
- **Model**: Pythia 6.9B
- **Loss**: Standard Supervised Fine-Tuning Loss
- **Hardware**: 3× H100 GPUs (80 GB VRAM each)
- **Training Time**: ~20 minutes  
  
  **Training Hyperparameters**:
- Batch Size: 64
- Eval Batch Size: 32
- Evaluation Frequency: Every 500 examples
- Gradient Accumulation Steps: 2  
  
  **Training Command**:
  ```python
  CUDA_VISIBLE_DEVICES=0,1,2 python -u train.py model=pythia69 datasets=[hs] loss=sft \
  exp_name=helpsteer2d-nosiy_sft_pythia69 gradient_accumulation_steps=2 batch_size=64 \
  eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16
  ```
-
- ## 2. 2D-DPO (Without Noise)
- **Dataset**: Original HelpSteer 2D dataset (clean)
- **Model**: Fine-tuned Pythia 6.9B from SFT phase
- **Training Hyperparameters**:
	- BETA: 0.1
	- Optimizer: RMSprop or AdamW
	- LR: 5e-7
	- Seed: 0
	- Batch Size: 16
	- Eval Batch Size: 8
	- Max Length: 512 (Prompt: 256)
	- Warmup Steps: 100
	- Gradient Accumulation: 2
	- Max Grad Norm: 10.0
	- Eval Every: 500 steps
	- Samples per Eval: 16
	- Sample During Eval: False
- **Model Precision**:
	- Policy Dtype: float32
	- Reference Dtype: float16
- **Paths & Logging**:
	- Model Archive:
	  `/nfs/ritik.ritik/Nosiy_2D-DPO/.cache/ritik.ritik/helpsteer2d-nosiy_sft_pythia69_2025-05-02_15-54-57_777221/LATEST/policy.pt`
	- Run Directory:
	  `/nfs/ritik.ritik/Nosiy_2D-DPO/2ddpo_pure_checkpoints`
- **Training Command**:
  ```python
  python 2ddpo_tp.py
  ```
- Compare logs at: [Weights & Biases](https://wandb.ai/ritik007/2d-pure-dpo_FSDP?nw=nwuserritik007)
- ## 3. 2D-DPO (With Noise)
- **Dataset**: Noisy HelpSteer 2D dataset
	- **Noise Injection**: Uniform noise added to `chosen_segment_scores` and `rejected_segment_scores`
- **Model & Training Setup**:
	- Identical architecture and hyperparameters as the clean 2D-DPO training
	- Used the same fine-tuned SFT model as initialization
	- Used to evaluate the robustness of 2D-DPO to scoring noise
- **Training Command**:
  ```python
  python 2ddpo_tp.py
  ```
- Compare logs at: [Weights & Biases](https://wandb.ai/ritik007/2d-noisy-dpo_TP?nw=nwuserritik007)

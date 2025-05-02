# Robust-2D-DPO
Direct Preference Optimization (DPO) aligns LLMs with human preferences but lacks granular scoring. 2D-DPO improves this with a two-dimensional scoring system. This work explores 2D-DPO’s advantages, addresses its sensitivity to label noise, and proposes a robust version, backed by proof and empirical results, with future research directions.

1) SFT Training:
- **Dataset**: `Noisy_2D-DPO.csv` (6,400 rows)  
  **Format**: _, instruction, chosen, rejected, chosen_segment_scores, rejected_segment_scores  
- **Training**:
- **Model**: Pythia 6.9B
- **Loss**: Supervised Fine-Tuning Loss
- **Batch Size**: 64
- **Eval Batch Size**: 32
- **Evaluation Frequency**: Every 500 examples
- **Training Time**: ~20 minutes
- **Hardware**: 3× H100 GPUs (80 GB VRAM each)
- **Training Command**:
  ```bash CUDA_VISIBLE_DEVICES=0,1,2 python -u train.py model=pythia69 datasets=[hs] loss=sft exp_name=helpsteer2d-nosiy_sft_pythia69 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 ```

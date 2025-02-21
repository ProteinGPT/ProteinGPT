export MODEL=Meta-Llama-3-8B-Instruct-hf/
mkdir -p minigpt4/ft/$MODEL
/home/ubuntu/miniconda3/envs/prot/bin/torchrun --master_port 28888 protein_gpt.py --cfg-path configs/train_instruction_tuning.yaml

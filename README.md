# ProteinGPT: Multimodal LLM for Protein Property Prediction and Structure Understanding

## Overview
ProteinGPT is a multi-modal protein chat system that allows users to upload protein sequences and/or structures for comprehensive protein analysis and interactive inquiries. It integrates protein sequence and structure encoders with a large language model (LLM) to generate contextually relevant and precise responses. 

## Features
- **Multimodal Input**: Accepts protein sequences (FASTA) and structures (PDB).
- **Advanced Encoding**: Utilizes pre-trained sequence and structure encoders for feature extraction.
- **LLM Integration**: Aligns protein representations with large language models for insightful responses.
- **Extensive Training Data**: Trained on a large-scale dataset of 132,092 proteins with annotations and optimized using instruction tuning with GPT-4o.
- **Modality Alignment & Instruction Tuning**: Two-stage training for enhanced understanding and response accuracy.

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.8+
- PyTorch
- CUDA (2 NVIDIA H100 80GB preferred)
- Conda

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/ProteinGPT/ProteinGPT
   cd ProteinGPT
   ```
2. Create and activate the environment:
   ```sh
   conda env create -f environment.yml
   conda activate proteingpt
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Training
To train ProteinGPT, run the following command:
```sh
bash train.sh
```
This script:
- Sets the base model to `Meta-Llama-3-8B-Instruct-hf`.
- Creates a directory for fine-tuning outputs.
- Launches training using `protein_gpt.py` with `configs/train_instruction_tuning.yaml`.

### Inference & Demo
To test the model interactively, run:
```sh
bash src/demo.sh
```
Or, to use the protein encoder module:
```sh
python demo_esm.py --cfg-path configs/evaluation.yaml  --gpu-id 0  --pdb $PATH_TO_STRUCTURE_EMBEDDING --seq $PATH_TO_SEQUENCE_EMBEDDING
```

## Directory Structure
```
ProteinGPT/
├── train.sh              # Training script
├── environment.yml       # Conda environment configuration
├── requirements.txt      # Python dependencies
├── src/
│   ├── protein_gpt.py   # Main model script
│   ├── demo.sh          # Demo script
│   ├── demo_esm.py      # Protein encoder demo
│   ├── esm/             # Sequence encoder module
│   ├── prompts/         # Prompt templates
│   ├── data/            # Training datasets
│   ├── configs/         # Configuration files
│   ├── minigpt4/        # Model-related files
├── LICENSE              # License file
├── README.md            # Guidance
```

## Model Architecture
ProteinGPT consists of:
1. **Protein Sequence Encoder**: Uses ESM-2 (`esm2_t36_3B_UR50D`).
2. **Protein Structure Encoder**: Uses an inverse folding model (`esm_if1_gvp4_t16_142M_UR50`).
3. **Projection Layer**: Aligns embeddings with the LLM.
4. **LLM Backbone**: Fine-tuned via instruction tuning.

## Dataset
ProteinGPT is trained on the **ProteinQA** dataset, derived from RCSB-PDB, containing:
- 132,092 protein entries
- Detailed annotations
- 20-30 property tags per protein
- 5-10 curated QA pairs

## Benchmark Results
ProteinGPT outperforms standard LLMs (GPT-4, LLaMA, Mistral) in protein-related tasks, achieving higher semantic similarity scores and improved factual accuracy in closed-ended QA experiments.

## Future Work
- **Hallucination Reduction**: Improve factual accuracy in responses.
- **Citations & References**: Integrate verifiable sources.
- **Expanded Training Data**: Incorporate additional protein knowledge bases.

## BibTeX
```
@article{xiao2024proteingpt,
  title={Proteingpt: Multimodal llm for protein property prediction and structure understanding},
  author={Xiao, Yijia and Sun, Edward and Jin, Yiqiao and Wang, Qifan and Wang, Wei},
  journal={arXiv preprint arXiv:2408.11363},
  year={2024}
}
```
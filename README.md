# Fine_Tuning
AI Model Development and Evaluation Pipeline

This repository contains a comprehensive pipeline for training, evaluating, and optimizing language models using state-of-the-art machine learning techniques. The project involves the following core steps:
# Data Preparation
# Model Training
# Model Evaluation

# Data_preparation.ipynb
This notebook focuses on preparing the dataset for fine-tuning. It utilizes the Hugging Face Transformers library and involves:
Loading and Tokenizing Data: Using the AutoTokenizer from Hugging Face to encode textual data.
Padding and Truncation: Managing input length using padding and truncation to ensure uniform input sizes.
Dataset Processing: Creating a fine-tuning dataset from a JSONL file, which includes questions and answers formatted into prompts.
Dataset Splitting: Splitting the dataset into training and test sets using a 90/10 ratio.
Dataset Upload: Demonstrating how to push datasets to the Hugging Face Hub for public or private use.

# Training_lab.ipynb
This notebook outlines the training process for fine-tuning a language model. It includes:
Model Selection: Using the EleutherAI/pythia-70m model from Hugging Face as the base model.
Data Tokenization: Tokenizing and splitting the dataset using the custom tokenize_and_split_data function.
Model Loading: Loading the AutoModelForCausalLM from Hugging Face and preparing it for training.
Training Configuration: Defining training parameters using the TrainingArguments class from Hugging Face, including learning rate, batch size, and number of training steps.
Training Execution: Using the Hugging Face Trainer API to execute the training process.
Inference Function: Implementing an inference function to generate predictions from the trained model.
Model Saving: Saving the fine-tuned model to a local directory for future use.
The notebook also explores different fine-tuned models, including a smaller model (pythia-70m) and a larger one (pythia-410m).

# Evaluation.ipynb
This notebook covers the evaluation process to assess the performance of the fine-tuned model. Key tasks include:
Model Loading: Loading the fine-tuned model from Hugging Face.
Inference and Comparison: Running predictions on the test dataset and comparing them with target answers.
Exact Match Calculation: Implementing a function to calculate exact matches between predicted and target answers.
Batch Evaluation: Evaluating a batch of test samples to calculate the overall accuracy of the model.
Evaluation Results: Generating a DataFrame of predictions and corresponding target answers for further analysis.
External Evaluation: Using the lm-evaluation-harness tool to benchmark the model on external datasets.

📊 Models Used

The following models are used in this project:
EleutherAI/pythia-70m: A 70-million parameter causal language model from Hugging Face.
lamini/lamini_docs_finetuned: A fine-tuned version of the base model trained on custom question-answer pairs.
pythia-410m: A larger 410-million parameter version of the Pythia model used for comparison.
These models are fine-tuned on custom datasets using the Hugging Face Transformers and Datasets libraries.

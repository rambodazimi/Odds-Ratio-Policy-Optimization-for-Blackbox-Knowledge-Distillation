# ORPO for Blackbox Knowledge Distillation

## **Author**: Rambod Azimi  

This repository contains an implementation of **Odds-Ratio Policy Optimization (ORPO)** for **Blackbox Knowledge Distillation (KD)** in Large Language Models (LLMs). The experiment explores whether ORPO fine-tuning can be adapted for blackbox KD settings where only teacher outputs (not logits) are available.

## **Overview**

### **What is ORPO?**
ORPO (Odds-Ratio Policy Optimization) is a fine-tuning technique designed to enhance the adaptability of LLMs for specific tasks by addressing limitations in traditional fine-tuning approaches. The key idea is to train on **triplets** of:

- **Prompt**: Input query or question
- **Chosen Response**: The best response selected by the model
- **Rejected Response**: A lower-quality response that the model should avoid

The goal is to maximize the **odds ratio** between the probabilities of the chosen response and the rejected response, improving the model's ability to generate high-quality outputs.

### **What is Blackbox KD?**
Blackbox KD (Knowledge Distillation) is a technique used to transfer knowledge from a large **teacher** model to a smaller **student** model when access to the teacher's logits is unavailable. Instead, the student learns from the **teacher's output responses** (soft labels). The training objective in this setting is defined by the **Negative Log-Likelihood (NLL) Loss**.

## **Implementation Details**

### **Dataset and Models**
- The experiment is conducted on the **MeQSum** dataset, which is designed for medical question summarization. The dataset consists of **1000 health-related questions** paired with summarized versions.
- The **teacher model** used in this experiment is **TinyLlama (1.1B)**, a small yet capable LLM model.
- The **student model** is **Flan-T5-base (250M)**, a lightweight model fine-tuned using ORPO principles.
- Both models are evaluated based on their ability to generate high-quality responses under blackbox KD constraints.

### **Training Process**
1. **Generating Triplet Data**: The dataset is processed to extract prompts, and responses are generated using both the teacher and student models. The best response is designated as the **chosen response**, while a lower-ranked response is set as the **rejected response**.
2. **Tokenization**: The triplet data is preprocessed and tokenized to be compatible with the student model.
3. **Fine-Tuning with ORPO**: The student model undergoes fine-tuning using a training objective that maximizes the odds ratio between the chosen and rejected responses. 
4. **Loss Computation**: A custom loss function incorporating **Negative Log-Likelihood (NLL) Loss** and **KL-Divergence** is used to guide the student model’s learning process.
5. **Model Optimization**: The model is trained with gradient-based optimization techniques to improve performance and minimize loss.

## **Experiment Description**

- **Task**: Experiment with fine-tuning using ORPO in the context of Blackbox KD for LLMs.
- **Dataset**: [MeQSum](https://github.com/abachaa/MeQSum) – a dataset for **medical question summarization**, consisting of **1000 health questions** with their corresponding summarized versions.
- **Teacher Model**: [TinyLlama (1.1B)](https://huggingface.co/TinyLlama/TinyLlama-1.1B)
- **Student Model**: [Flan-T5-base (250M)](https://huggingface.co/google/flan-t5-base)
- **Evaluation Metrics**: BLEU and ROUGE scores
- **Baselines**: Student model with Zero-shot and Few-shot inference, Teacher model with Zero-shot and Few-shot inference.

## **Installation**

To run this notebook, install the required dependencies:

```bash
pip install transformers datasets torch rouge_score nltk trl
```

## **Conclusion**

This implementation explores the feasibility of ORPO for blackbox knowledge distillation. The key takeaways are:

- ORPO improves model adaptability by leveraging triplet-based training.
- Blackbox KD is feasible even when only **output responses** (not logits) are available from the teacher.
- Evaluation with BLEU/ROUGE scores shows performance improvements in fine-tuned student models.

## **Acknowledgments**
- The MeQSum dataset
- Hugging Face models (TinyLlama, Flan-T5)
- ORPO Training framework


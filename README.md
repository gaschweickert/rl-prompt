# Prompt Optimization with Reinforcement Learning for Few-Shot Text Classification

## Overview
This project replicates and adapts the RLPrompt method from "Optimizing Discrete Text Prompts with Reinforcement Learning" (Deng et al.) for few-shot text classification. It explores using Deep Q-Learning (DQN) instead of Soft Q-Learning for prompt generation.

## Methodology
- Task: Few-shot text classification (16 examples per class)
- Approach: Discrete prompt optimization via reinforcement learning
- Model: DistilRoBERTa (82M parameters) for classification, DistilGPT2 for prompt generation
- RL Algorithm: Deep Q-Learning (DQN)
- Prompt Length: T = 5 tokens
- Datasets: SST-2 and Yelp (binary sentiment analysis)

## Key Components
1. Frozen Transformer + Task-Specific MLP + Frozen LM Head architecture
2. Input-specific prompt generation using DistilGPT2
3. DQN with epsilon-greedy exploration strategy
4. Piece-wise reward function for stable RL training

## Experiments
- Compared RLPrompt (DQN) against manual prompts and fine-tuning baselines
- Evaluated on SST-2 and Yelp datasets
- Metrics: Accuracy on 100-sample hold-out test set

## Results
- RLPrompt (DQN) average accuracy: 63.75%
- Manual prompts average accuracy: 72.75%
- Fine-tuning average accuracy: 77.25%

## Challenges & Observations
- Convergence to adversarial prompts (e.g., Unicode replacement characters)
- Overfitting issues when mitigating adversarial prompts
- Instability and sensitivity to hyperparameters in DQN approach

## Future Work
- Implement Soft Q-Learning as in the original paper
- Explore other policy gradient algorithms (e.g., REINFORCE)
- Investigate methods to improve prompt interpretability
- Address environmental concerns of computationally expensive training

## Technologies
- PyTorch
- Hugging Face Transformers
- Reinforcement Learning (DQN)

## Ethical Considerations
- Potential for bias propagation and amplification
- Environmental impact of computationally intensive training
- Interpretability challenges with generated prompts

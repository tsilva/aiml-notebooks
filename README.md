<div align="center">
  <img src="logo.png" alt="aiml-notebooks" width="512"/>

  # aiml-notebooks

  [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB.svg)](https://www.python.org/)
  [![uv](https://img.shields.io/badge/uv-package%20manager-5C4EE5.svg)](https://docs.astral.sh/uv/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

  **📚 AI/ML Jupyter notebooks for learning deep learning concepts 🧠**

  [Learning Path](#-learning-path) · [Quick Start](#-quick-start) · [Learning Tracks](#-learning-tracks)
</div>

---

## Overview

This repository provides **100+ hands-on Jupyter notebooks** covering machine learning and deep learning. The curriculum starts with a shared foundation, then branches into focused domains like transformers, computer vision, generative models, and reinforcement learning.

**Why this repository?**
- **Conceptual ordering**: Core tiers are sequenced by prerequisites, then later tiers branch into domain-specific tracks
- **Implementation-first**: Learn by building everything from scratch before using frameworks
- **Broad coverage**: From linear algebra to transformers, diffusion, and reinforcement learning

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone https://github.com/tsilva/aiml-notebooks.git
cd aiml-notebooks
uv sync
git config core.hooksPath .githooks
```

The tracked pre-commit hook strips outputs and execution counts from staged `.ipynb` files before each commit.

### Run Jupyter Lab

```bash
uv run jupyter lab
```

## Learning Tracks

Choose a focused path based on your goals:

| Track | Tiers | Focus |
|-------|-------|-------|
| **Fast Track to LLMs** | 1 → 2 → 3 → 7 → 8 → 9 → 11 | Quickest path to transformer and LLM foundations |
| **Computer Vision** | 1 → 2 → 3 → 6 → 7 → 10 → 11 → 13 → 16 → 17 | CNNs, ViTs, image generation, and interpretability |
| **Classical ML Mastery** | 1 → 2 → 3 → 4 → 5 → 6 | Strong foundations before deep learning |
| **Reinforcement Learning** | 1 → 2 → 3 → 5 → 5A → 7 → 19 | RL foundations, deep RL, and the evaluation discipline needed to train them well |
| **NLP Applications** | 1 → 2 → 3 → 7 → 8 → 9 → 11 → 14 → 12 | Language modeling, transformers, downstream NLP tasks, then advanced LLM systems |
| **Complete Curriculum** | 1 → 12 core, then 13-21 by interest | Comprehensive understanding across the core and advanced branches (~180-220 hours) |

---

## Learning Path

**Legend:** ✅ Completed · 🔴 Advanced · 🔄 In Progress · 🔜 Planned

**Structure note:** Tiers 1-12 form the core prerequisite spine. Tiers 13-21 are advanced branches and electives that can be taken after the relevant foundations rather than as one strict linear chain.

### TIER 1: Numerical Computing

Learn to work with arrays and tensors—the data structures of ML.

| Notebook | Description |
|----------|-------------|
| ✅ [numpy-fundamentals](notebooks/numpy-fundamentals.ipynb) | NumPy arrays, indexing, broadcasting, and operations |
| [tensors-operations](notebooks/tensors-operations.ipynb) | PyTorch tensor operations: shapes, broadcasting, indexing |
| [tensor-multiplication](notebooks/tensor-multiplication.ipynb) | Matrix multiplication and broadcasting in detail |

### TIER 2: Mathematical Foundations

The math you need—linear algebra, calculus, and probability.

| Notebook | Description |
|----------|-------------|
| [linear-algebra](notebooks/linear-algebra.ipynb) | Vectors, matrices, dot products, norms, eigenvalues |
| [calculus-refresher](notebooks/calculus-refresher.ipynb) | Derivatives, chain rule, partial derivatives, gradients |
| [probability-distributions](notebooks/probability-distributions.ipynb) | Discrete and continuous distributions |
| [bayesian-ml-basics](notebooks/bayesian-ml-basics.ipynb) | Bayes' theorem, priors, posteriors, Bayesian linear regression |

### TIER 3: Core ML Components

Essential building blocks used in every ML model.

| Notebook | Description |
|----------|-------------|
| [data-normalization](notebooks/data-normalization.ipynb) | Min-max scaling vs standardization, avoiding data leakage |
| [loss-functions](notebooks/loss-functions.ipynb) | MSE, MAE, cross-entropy, and hinge loss |
| [activation-functions](notebooks/activation-functions.ipynb) | ReLU, sigmoid, tanh and gradient flow effects |
| [metrics](notebooks/metrics.ipynb) | Accuracy, precision, recall, F1, ROC-AUC |
| [kl-divergence](notebooks/kl-divergence.ipynb) | Information theory: surprisal, entropy, cross-entropy, KL divergence |

### TIER 4: Classical ML Algorithms

Traditional machine learning—master these before deep learning.

| Notebook | Description |
|----------|-------------|
| [linear-regression](notebooks/linear-regression.ipynb) | Gradient descent vs closed-form solution |
| [logistic-regression](notebooks/logistic-regression.ipynb) | Binary classification and decision boundaries |
| [softmax-multiclass](notebooks/softmax-multiclass.ipynb) | Softmax, one-hot encoding, categorical cross-entropy |
| [decision-trees](notebooks/decision-trees.ipynb) | Recursive splitting, Gini, entropy, information gain |
| [naive-bayes](notebooks/naive-bayes.ipynb) | Gaussian, Multinomial, Bernoulli NB, Laplace smoothing |
| [support-vector-machines](notebooks/support-vector-machines.ipynb) | Maximum margin, kernel trick, C and gamma |
| [kmeans-from-scratch](notebooks/kmeans-from-scratch.ipynb) | K-means clustering, k-means++, elbow method |
| [gaussian-mixture-models](notebooks/gaussian-mixture-models.ipynb) | GMMs and Expectation-Maximization |

### TIER 5: Generalization & Model Selection

Build models that work on new data.

| Notebook | Description |
|----------|-------------|
| [bias-variance-overfitting](notebooks/bias-variance-overfitting.ipynb) | Bias-variance tradeoff, learning curves |
| [train-test-val-split](notebooks/train-test-val-split.ipynb) | Splits, data leakage prevention, temporal splits |
| [cross-validation](notebooks/cross-validation.ipynb) | K-Fold, Stratified K-Fold, Time Series CV |
| [regularization](notebooks/regularization.ipynb) | L1, L2, dropout, early stopping |
| [confidence-intervals](notebooks/confidence-intervals.ipynb) | Bootstrap, hypothesis testing |
| [hyperparameter-tuning](notebooks/hyperparameter-tuning.ipynb) | Grid, random, and Bayesian optimization |
| [ensemble-methods](notebooks/ensemble-methods.ipynb) | Bagging, Random Forests, boosting, stacking |

### TIER 5A: RL Foundations

Core RL concepts that don't require deep learning.

| Notebook | Description |
|----------|-------------|
| [rl-q-learning](notebooks/rl-q-learning.ipynb) | Q-learning, MDPs, Bellman equation, ε-greedy |
| [mcts-simple](notebooks/mcts-simple.ipynb) | Monte Carlo Tree Search with UCB1 |
| [rl-model-based](notebooks/rl-model-based.ipynb) | Tabular world models, Dyna, prioritized sweeping |

### TIER 6: Data Engineering

Making the most of your data.

| Notebook | Description |
|----------|-------------|
| [dimensionality-reduction](notebooks/dimensionality-reduction.ipynb) | PCA, t-SNE, UMAP |
| [feature-engineering](notebooks/feature-engineering.ipynb) | Encoding, binning, and feature pipeline design |
| [data-augmentation](notebooks/data-augmentation.ipynb) | Augmentation for image data |
| [imbalanced-data](notebooks/imbalanced-data.ipynb) | Class weights, resampling, SMOTE |

### TIER 7: Deep Learning Foundations

Neural networks from scratch—the core of modern ML.

| Notebook | Description |
|----------|-------------|
| [neural-network-fundamentals](notebooks/neural-network-fundamentals.ipynb) | Neurons, layers, Universal Approximation Theorem |
| [computational-graphs](notebooks/computational-graphs.ipynb) | DAGs and the chain rule for backpropagation |
| [z2h-01-backprop](notebooks/z2h-01-backprop.ipynb) | Backprop from scratch: micrograd-style autograd |
| [pytorch-fundamentals](notebooks/pytorch-fundamentals.ipynb) | Autograd, nn.Parameter, nn.Module, training loop |
| ✅ [mini-batch-gradient-descent](notebooks/mini-batch-gradient-descent.ipynb) | Full-batch, mini-batch, and SGD |
| ✅ [optimizers](notebooks/optimizers.ipynb) | SGD, Momentum, RMSprop, Adam |
| ✅ [learning-rate-schedules](notebooks/learning-rate-schedules.ipynb) | Step decay, cosine annealing, warmup |
| [z2h-04-optimization-pt1](notebooks/z2h-04-optimization-pt1.ipynb) | Xavier/Kaiming initialization, activation flow |
| [gradient-flow-and-stabilization](notebooks/gradient-flow-and-stabilization.ipynb) | Gradient flow, clipping, and normalization |
| [batch-normalization](notebooks/batch-normalization.ipynb) | BatchNorm for stable training |
| ✅ [layer-normalization](notebooks/layer-normalization.ipynb) | LayerNorm for RNNs and transformers |
| [debugging-neural-networks](notebooks/debugging-neural-networks.ipynb) | Systematic debugging and sanity checks |

### TIER 8: Text & Embedding Foundations

Essential text processing before language models.

| Notebook | Description |
|----------|-------------|
| [tokenization-methods](notebooks/tokenization-methods.ipynb) | Character, word, BPE, WordPiece |
| ✅ [embeddings](notebooks/embeddings.ipynb) | Learned vector representations |
| 🔄 [word2vec-from-scratch](notebooks/word2vec-from-scratch.ipynb) | Skip-gram with negative sampling |

### TIER 9: Sequence Modeling Foundations

Early neural sequence models before full transformers.

| Notebook | Description |
|----------|-------------|
| [z2h-02-bigram-lm](notebooks/z2h-02-bigram-lm.ipynb) | Character-level bigram language models |
| [z2h-03-mlp-lm](notebooks/z2h-03-mlp-lm.ipynb) | MLP for character-level language modeling |
| [rnn-from-scratch](notebooks/rnn-from-scratch.ipynb) | RNN with BPTT |
| [lstm-from-scratch](notebooks/lstm-from-scratch.ipynb) | LSTM gates and vanishing gradients |
| [gru-from-scratch](notebooks/gru-from-scratch.ipynb) | GRU: simpler gated architecture |
| [seq2seq-with-attention](notebooks/seq2seq-with-attention.ipynb) | Encoder-decoder sequence modeling with attention |

### TIER 10: Convolutional Neural Networks

Computer vision fundamentals.

| Notebook | Description |
|----------|-------------|
| [tensors-convolution](notebooks/tensors-convolution.ipynb) | Filters, padding, stride, dilation |
| [pooling-operations](notebooks/pooling-operations.ipynb) | MaxPool, AvgPool, GlobalPool |
| [lenet-limitations](notebooks/lenet-limitations.ipynb) | LeNet-5: the first successful CNN |
| [alexnet-breakthrough](notebooks/alexnet-breakthrough.ipynb) | AlexNet: the ImageNet breakthrough |
| [vgg-depth-uniformity](notebooks/vgg-depth-uniformity.ipynb) | VGG: uniform 3×3 architecture |
| [inception-multiscale](notebooks/inception-multiscale.ipynb) | Inception: multi-scale parallel processing |
| [resnet-skip-connections](notebooks/resnet-skip-connections.ipynb) | ResNet: solving the degradation problem |
| [mobilenet-efficient-cnns](notebooks/mobilenet-efficient-cnns.ipynb) | Depthwise separable convolutions |
| 🔴 [efficientnet-compound-scaling](notebooks/efficientnet-compound-scaling.ipynb) | Compound scaling with NAS |
| [transfer-learning](notebooks/transfer-learning.ipynb) | Feature extraction vs fine-tuning |
| [unet-architecture](notebooks/unet-architecture.ipynb) | U-Net for semantic segmentation |
| [object-detection-yolo](notebooks/object-detection-yolo.ipynb) | YOLO: bounding boxes, anchor boxes, NMS |

### TIER 11: Attention & Transformer Foundations

Modern sequence modeling—the revolution.

| Notebook | Description |
|----------|-------------|
| ✅ [attention-mechanism](notebooks/attention-mechanism.ipynb) | Scaled dot-product attention: Q, K, V |
| [positional-encodings](notebooks/positional-encodings.ipynb) | Sinusoidal, Learned, RoPE, ALiBi |
| [transformer-from-scratch](notebooks/transformer-from-scratch.ipynb) | Full Transformer architecture |
| ✅ [gpt-architecture](notebooks/gpt-architecture.ipynb) | GPT decoder-only transformers, autoregressive generation, and greedy/temperature/top-k/top-p sampling |
| ✅ [bert-architecture](notebooks/bert-architecture.ipynb) | BERT encoder, masked LM |
| 🔄 [vision-transformers](notebooks/vision-transformers.ipynb) | ViT: patch embeddings, 2D positional encodings |

### TIER 12: LLM Adaptation, Inference & Systems

Advanced transformer adaptation, inference, and deployment topics after Tier 11. This tier now includes the fine-tuning bridge between transformer foundations, PEFT, and downstream NLP applications.

| Notebook | Description |
|----------|-------------|
| [transformer-fine-tuning-basics](notebooks/transformer-fine-tuning-basics.ipynb) | Full fine-tuning vs frozen backbones, task heads, supervised fine-tuning flow, evaluation/overfitting pitfalls, and when PEFT becomes necessary |
| [lora-peft](notebooks/lora-peft.ipynb) | LoRA, QLoRA, and PEFT tradeoffs across full fine-tuning, adapters, and prompt tuning |
| [gpu-hardware-basics](notebooks/gpu-hardware-basics.ipynb) | CUDA cores, memory hierarchy, Tensor Cores, and why hardware shapes LLM systems |
| 🔴 [flash-attention](notebooks/flash-attention.ipynb) | Memory-efficient attention with tiling |
| [kv-caching](notebooks/kv-caching.ipynb) | KV caching for efficient inference |
| [speculative-decoding](notebooks/speculative-decoding.ipynb) | Accelerating LLM inference |

### TIER 13: Representation Learning

Learning without labels and useful representations.

| Notebook | Description |
|----------|-------------|
| ✅ [autoencoders](notebooks/autoencoders.ipynb) | Encoder-decoder for compression |
| 🔄 [sparse-autoencoders](notebooks/sparse-autoencoders.ipynb) | Sparse AEs with KL/L1 for interpretability |
| [self-supervised-learning](notebooks/self-supervised-learning.ipynb) | Overview of rotation, jigsaw, masking, and SimCLR |
| 🔄 [contrastive-learning](notebooks/contrastive-learning.ipynb) | Deep dive on InfoNCE loss and SimCLR-style SSL |

### TIER 14: NLP Applications

Practical NLP tasks end-to-end.

| Notebook | Description |
|----------|-------------|
| [classification-text](notebooks/classification-text.ipynb) | Sentiment: TF-IDF, LSTM, BERT |
| [denoising-text](notebooks/denoising-text.ipynb) | Text correction, CER/WER |
| [ner-token-classification](notebooks/ner-token-classification.ipynb) | NER with BIO tagging, BiLSTM-CRF sequence constraints, and Viterbi decoding |
| [qa-extractive](notebooks/qa-extractive.ipynb) | Extractive QA with BiDAF, BERT, answer span decoding, and retrieval-backed QA context |
| [retrieval-fundamentals](notebooks/retrieval-fundamentals.ipynb) | Sparse vs dense retrieval, BM25, ANN/vector indexes, chunking tradeoffs, and retrieval metrics like recall@k / MRR / nDCG |
| [rag-retrieval-augmented](notebooks/rag-retrieval-augmented.ipynb) | RAG with chunking, vector search, dense vs sparse retrieval, and BM25 |

### TIER 15: Audio Processing

Speech and audio applications.

| Notebook | Description |
|----------|-------------|
| [audio-processing](notebooks/audio-processing.ipynb) | FFT, spectrograms, mel, MFCCs |
| [speech-recognition](notebooks/speech-recognition.ipynb) | ASR with CTC loss |

### TIER 16: Generative Models

Learning to generate images and data.

| Notebook | Description |
|----------|-------------|
| [denoising-image](notebooks/denoising-image.ipynb) | Image denoising with conv AEs |
| [vae](notebooks/vae.ipynb) | VAE: reparameterization trick |
| [vqvae](notebooks/vqvae.ipynb) | VQ-VAE: discrete latent spaces |
| [gan-fundamentals](notebooks/gan-fundamentals.ipynb) | GANs: minimax game, mode collapse |
| [neural-style-transfer](notebooks/neural-style-transfer.ipynb) | Artistic image generation with CNNs |
| [diffusion-models](notebooks/diffusion-models.ipynb) | DDPM and DDIM sampling |
| [latent-diffusion](notebooks/latent-diffusion.ipynb) | Latent Diffusion and Stable Diffusion |

### TIER 17: Model Optimization & Interpretability

Making models efficient and understandable.

| Notebook | Description |
|----------|-------------|
| [model-interpretability](notebooks/model-interpretability.ipynb) | LIME, SHAP, integrated gradients |
| [grad-cam-visualization](notebooks/grad-cam-visualization.ipynb) | Visualizing CNN decisions |
| [knowledge-distillation](notebooks/knowledge-distillation.ipynb) | Teacher-student training |
| [model-pruning](notebooks/model-pruning.ipynb) | Unstructured/structured pruning |
| [int8-fp16-quantization](notebooks/int8-fp16-quantization.ipynb) | INT8/FP16 quantization techniques |
| [1bit-neural-networks](notebooks/1bit-neural-networks.ipynb) | Binary/ternary quantization |

### TIER 18: Advanced Architectures

Beyond standard networks.

| Notebook | Description |
|----------|-------------|
| [clip-architecture](notebooks/clip-architecture.ipynb) | CLIP dual encoders, zero-shot |
| [mixture-of-experts](notebooks/mixture-of-experts.ipynb) | Sparse MoE, gating networks |
| [neural-turing-machines](notebooks/neural-turing-machines.ipynb) | NTMs: external differentiable memory |
| 🔴 [capsule-networks](notebooks/capsule-networks.ipynb) | CapsNets, dynamic routing |
| 🔴 [graph-neural-networks](notebooks/graph-neural-networks.ipynb) | GNNs and message passing |
| 🔴 [neural-odes](notebooks/neural-odes.ipynb) | Continuous-depth networks as ODEs |
| 🔴 [energy-based-neural-networks](notebooks/energy-based-neural-networks.ipynb) | Hopfield Networks, RBMs |

### TIER 19: Deep Reinforcement Learning

Learning through interaction.

| Notebook | Description |
|----------|-------------|
| [rl-deep-q-networks](notebooks/rl-deep-q-networks.ipynb) | DQN with experience replay |
| [rl-policy-gradients](notebooks/rl-policy-gradients.ipynb) | REINFORCE, Actor-Critic, PPO |
| [rl-advanced-policy-methods](notebooks/rl-advanced-policy-methods.ipynb) | A2C/A3C, SAC, TD3 |

### TIER 20: Fascinating Phenomena

Research findings that challenge intuitions.

| Notebook | Description |
|----------|-------------|
| [adversarial-robustness](notebooks/adversarial-robustness.ipynb) | FGSM/PGD attacks, adversarial training |
| 🔴 [double-descent](notebooks/double-descent.ipynb) | Test error decreasing beyond interpolation |
| 🔴 [grokking](notebooks/grokking.ipynb) | Sudden generalization after overfitting |

### TIER 21: Alternative Learning Paradigms

Beyond standard supervised learning.

| Notebook | Description |
|----------|-------------|
| [meta-learning-few-shot](notebooks/meta-learning-few-shot.ipynb) | MAML for few-shot learning |
| [active-learning](notebooks/active-learning.ipynb) | Uncertainty, QBC, diversity sampling |
| [curriculum-learning](notebooks/curriculum-learning.ipynb) | Training from easy to hard |
| [multi-task-learning](notebooks/multi-task-learning.ipynb) | Hard/soft parameter sharing |
| [continual-learning](notebooks/continual-learning.ipynb) | Avoiding catastrophic forgetting |
| 🔴 [forward-forward](notebooks/forward-forward.ipynb) | Layer-local learning without backprop |
| [neuroevolution-tic-tac-toe](notebooks/neuroevolution-tic-tac-toe.ipynb) | Genetic algorithms for neural nets |

### BONUS: Fun Projects

| Notebook | Description |
|----------|-------------|
| [chip8-emulator](notebooks/chip8-emulator.ipynb) | CHIP-8 emulator: fetch-decode-execute |

---

## Advanced Usage

### Hyperparameter Sweeps

```bash
uv run python sweep.py path/to/config.yaml notebooks/your-notebook.ipynb --count 10
```

Use any W&B sweep YAML that matches the parameters exposed by your notebook.

### Shared Library

```python
from aiml_notebooks import CharacterTokenizer, create_dataset, create_dataloaders, get_device, set_seed

%load_ext autoreload
%autoreload 2  # Hot reload during development
```

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
  <sub>Total estimated time for complete curriculum: ~180-220 hours</sub>
</div>

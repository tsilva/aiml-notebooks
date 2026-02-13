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

This repository provides **100+ hands-on Jupyter notebooks** covering machine learning and deep learning. Each notebook builds deep intuitions through progressive implementation—from NumPy fundamentals to transformers, diffusion models, and reinforcement learning.

**Why this repository?**
- **Conceptual ordering**: Notebooks are sequenced by prerequisites, not topics—each tier builds on previous tiers
- **Implementation-first**: Learn by building everything from scratch before using frameworks
- **Complete coverage**: From linear algebra to GPT-2, from decision trees to RLHF

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone https://github.com/tsilva/aiml-notebooks.git
cd aiml-notebooks
uv sync
```

### Run Jupyter Lab

```bash
uv run jupyter lab
```

## Learning Tracks

Choose a focused path based on your goals:

| Track | Tiers | Focus |
|-------|-------|-------|
| **Fast Track to LLMs** | 1 → 2 → 3 → 6 → 7 → 8 → 11 → 12 → 13 | Quickest path to understanding modern language models |
| **Computer Vision** | 1 → 2 → 3 → 6 → 7 → 9 → 10 → 15 | CNNs, architectures, and image generation |
| **Classical ML Mastery** | 1 → 2 → 3 → 4 → 5 → 6 | Strong foundations before deep learning |
| **Reinforcement Learning** | 1 → 2 → 3 → 4 → 5A → 6 → 7 → 19 | Decision-making and game-playing agents |
| **Complete Curriculum** | All tiers in order | Comprehensive understanding (~180-220 hours) |

---

## Learning Path

**Legend:** ✅ Completed · 🔴 Advanced · 🔄 In Progress

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

### TIER 6: Data Engineering

Making the most of your data.

| Notebook | Description |
|----------|-------------|
| [dimensionality-reduction](notebooks/dimensionality-reduction.ipynb) | PCA, t-SNE, UMAP |
| [feature-engineering](notebooks/feature-engineering.ipynb) | Encoding, missing values, feature creation |
| [data-augmentation](notebooks/data-augmentation.ipynb) | Augmentation for images and text |
| [imbalanced-data](notebooks/imbalanced-data.ipynb) | Class weights, resampling, SMOTE |

### TIER 7: Deep Learning Foundations

Neural networks from scratch—the core of modern ML.

| Notebook | Description |
|----------|-------------|
| [gpu-hardware-basics](notebooks/gpu-hardware-basics.ipynb) | CUDA cores, memory hierarchy, Tensor Cores |
| [neural-network-fundamentals](notebooks/neural-network-fundamentals.ipynb) | Neurons, layers, Universal Approximation Theorem |
| [computational-graphs](notebooks/computational-graphs.ipynb) | DAGs and the chain rule for backpropagation |
| [z2h-01-backprop](notebooks/z2h-01-backprop.ipynb) | Backprop from scratch: micrograd-style autograd |
| [pytorch-fundamentals](notebooks/pytorch-fundamentals.ipynb) | Autograd, nn.Parameter, nn.Module, training loop |
| ✅ [learning-xor-with-mlp](notebooks/learning-xor-with-mlp.ipynb) | MLP for non-linearly separable problems |
| ✅ [mini-batch-gradient-descent](notebooks/mini-batch-gradient-descent.ipynb) | Full-batch, mini-batch, and SGD |
| ✅ [optimizers](notebooks/optimizers.ipynb) | SGD, Momentum, RMSprop, Adam |
| ✅ [learning-rate-schedules](notebooks/learning-rate-schedules.ipynb) | Step decay, cosine annealing, warmup |
| [gradient-flow-and-stabilization](notebooks/gradient-flow-and-stabilization.ipynb) | Gradient flow analysis |
| [gradient-stabilization](notebooks/gradient-stabilization.ipynb) | Gradient clipping and normalization |
| [batch-normalization](notebooks/batch-normalization.ipynb) | BatchNorm for stable training |
| ✅ [layer-normalization](notebooks/layer-normalization.ipynb) | LayerNorm for RNNs and transformers |
| [z2h-04-optimization-pt1](notebooks/z2h-04-optimization-pt1.ipynb) | Xavier/Kaiming initialization, activation flow |
| [debugging-neural-networks](notebooks/debugging-neural-networks.ipynb) | Systematic debugging and sanity checks |

### TIER 8: Text & Embedding Foundations

Essential text processing before language models.

| Notebook | Description |
|----------|-------------|
| [tokenization-methods](notebooks/tokenization-methods.ipynb) | Character, word, BPE, WordPiece |
| ✅ [embeddings](notebooks/embeddings.ipynb) | Learned vector representations |
| 🔄 [word2vec-from-scratch](notebooks/word2vec-from-scratch.ipynb) | Skip-gram with negative sampling |

### TIER 9: First Neural Networks

Simple neural networks for language modeling.

| Notebook | Description |
|----------|-------------|
| [z2h-02-bigram-lm](notebooks/z2h-02-bigram-lm.ipynb) | Character-level bigram language models |
| [z2h-03-mlp-lm](notebooks/z2h-03-mlp-lm.ipynb) | MLP for character-level language modeling |

### TIER 10: Convolutional Neural Networks

Computer vision fundamentals.

| Notebook | Description |
|----------|-------------|
| [tensors-convolution](notebooks/tensors-convolution.ipynb) | Filters, padding, stride, dilation |
| [pooling-operations](notebooks/pooling-operations.ipynb) | MaxPool, AvgPool, GlobalPool |
| [cnn-architectures-evolution](notebooks/cnn-architectures-evolution.ipynb) | LeNet → AlexNet → VGG → Inception evolution |
| [lenet-limitations](notebooks/lenet-limitations.ipynb) | LeNet-5: the first successful CNN |
| [alexnet-breakthrough](notebooks/alexnet-breakthrough.ipynb) | AlexNet: the ImageNet breakthrough |
| [vgg-depth-uniformity](notebooks/vgg-depth-uniformity.ipynb) | VGG: uniform 3×3 architecture |
| [inception-multiscale](notebooks/inception-multiscale.ipynb) | Inception: multi-scale parallel processing |
| [resnet-skip-connections](notebooks/resnet-skip-connections.ipynb) | ResNet: solving the degradation problem |
| ✅ [why-residual-connections-work](notebooks/why-residual-connections-work.ipynb) | Deep dive into skip connections |
| [mobilenet-efficient-cnns](notebooks/mobilenet-efficient-cnns.ipynb) | Depthwise separable convolutions |
| 🔴 [efficientnet-compound-scaling](notebooks/efficientnet-compound-scaling.ipynb) | Compound scaling with NAS |
| [classification-image](notebooks/classification-image.ipynb) | CNN/MLP on CIFAR-10, MNIST |
| [transfer-learning](notebooks/transfer-learning.ipynb) | Feature extraction vs fine-tuning |
| [neural-style-transfer](notebooks/neural-style-transfer.ipynb) | Artistic image generation with CNNs |
| [unet-architecture](notebooks/unet-architecture.ipynb) | U-Net for semantic segmentation |
| [grad-cam-visualization](notebooks/grad-cam-visualization.ipynb) | Visualizing CNN decisions |
| [object-detection-yolo](notebooks/object-detection-yolo.ipynb) | YOLO: bounding boxes, anchor boxes, NMS |

### TIER 11: Recurrent Neural Networks

Sequential data and memory.

| Notebook | Description |
|----------|-------------|
| [rnn-from-scratch](notebooks/rnn-from-scratch.ipynb) | RNN with BPTT |
| [lstm-from-scratch](notebooks/lstm-from-scratch.ipynb) | LSTM gates and vanishing gradients |
| [gru-from-scratch](notebooks/gru-from-scratch.ipynb) | GRU: simpler gated architecture |
| ✅ [rnn-binary-operations](notebooks/rnn-binary-operations.ipynb) | RNN learning binary logic operations |
| ✅ [rwkv4-from-scratch](notebooks/rwkv4-from-scratch.ipynb) | RWKV-4: bridging RNNs and Transformers |
| ✅ [rwkv7-from-scratch](notebooks/rwkv7-from-scratch.ipynb) | RWKV-7 "Goose": expressive state dynamics |
| [mamba1-gentle-intro](notebooks/mamba1-gentle-intro.ipynb) ⭐ | Gentle introduction to Mamba from RNNs |
| [mamba1-from-scratch](notebooks/mamba1-from-scratch.ipynb) | Mamba: selective state space models |
| [time-series-forecasting](notebooks/time-series-forecasting.ipynb) | MLP, LSTM, Transformer for time series |
| [wip-linear-rnns](notebooks/wip-linear-rnns.ipynb) | Linear RNNs and parallel scan |

### TIER 12: Attention & Transformers

Modern sequence modeling—the revolution.

| Notebook | Description |
|----------|-------------|
| ✅ [attention-mechanism](notebooks/attention-mechanism.ipynb) | Scaled dot-product attention: Q, K, V |
| [seq2seq-with-attention](notebooks/seq2seq-with-attention.ipynb) | Encoder-decoder with attention |
| [positional-encodings](notebooks/positional-encodings.ipynb) | Sinusoidal, Learned, RoPE, ALiBi |
| [transformer-from-scratch](notebooks/transformer-from-scratch.ipynb) | Full Transformer architecture |
| ✅ [gpt-architecture](notebooks/gpt-architecture.ipynb) | GPT decoder-only, causal masking |
| ✅ [bert-architecture](notebooks/bert-architecture.ipynb) | BERT encoder, masked LM |
| [gpt2-from-scratch](notebooks/gpt2-from-scratch.ipynb) | GPT-2 in raw PyTorch (educational) |
| [gpt2-keras-jax](notebooks/gpt2-keras-jax.ipynb) | GPT-2 with Keras/JAX backend |
| [wip-gpt2-time-series](notebooks/wip-gpt2-time-series.ipynb) | GPT-2 for time series forecasting |
| 🔄 [vision-transformers](notebooks/vision-transformers.ipynb) | ViT: patch embeddings, 2D positional encodings |

### TIER 13: Efficient Transformers

Making transformers fast and deployable.

| Notebook | Description |
|----------|-------------|
| 🔴 [flash-attention](notebooks/flash-attention.ipynb) | Memory-efficient attention with tiling |
| [kv-caching](notebooks/kv-caching.ipynb) | KV caching for efficient inference |
| [speculative-decoding](notebooks/speculative-decoding.ipynb) | Accelerating LLM inference |
| [int8-fp16-quantization](notebooks/int8-fp16-quantization.ipynb) | INT8/FP16 quantization techniques |
| [lora-peft](notebooks/lora-peft.ipynb) | LoRA and parameter-efficient fine-tuning |

### TIER 14: Representation Learning

Learning without labels and useful representations.

| Notebook | Description |
|----------|-------------|
| ✅ [autoencoders](notebooks/autoencoders.ipynb) | Encoder-decoder for compression |
| 🔄 [sparse-autoencoders](notebooks/sparse-autoencoders.ipynb) | Sparse AEs with KL/L1 for interpretability |
| 🔄 [siamese-networks](notebooks/siamese-networks.ipynb) | Twin networks with contrastive loss |
| 🔄 [contrastive-learning](notebooks/contrastive-learning.ipynb) | InfoNCE loss, SimCLR-style SSL |
| [self-supervised-learning](notebooks/self-supervised-learning.ipynb) | Rotation, Jigsaw, MAE, SimCLR, BERT masking |

### TIER 15: NLP Applications

Practical NLP tasks end-to-end.

| Notebook | Description |
|----------|-------------|
| [classification-text](notebooks/classification-text.ipynb) | Sentiment: TF-IDF, LSTM, BERT |
| [generation-text](notebooks/generation-text.ipynb) | Text generation and sampling strategies |
| [reconstruction-text](notebooks/reconstruction-text.ipynb) | Seq2seq autoencoders |
| [denoising-text](notebooks/denoising-text.ipynb) | Text correction, CER/WER |
| [ner-token-classification](notebooks/ner-token-classification.ipynb) | NER with BIO tagging, BiLSTM-CRF, BERT |
| [qa-extractive](notebooks/qa-extractive.ipynb) | Extractive QA with BiDAF and BERT |
| [rag-retrieval-augmented](notebooks/rag-retrieval-augmented.ipynb) | RAG: chunking, embeddings, vector search |
| [prompt-engineering-llms](notebooks/prompt-engineering-llms.ipynb) | Zero-shot, few-shot, CoT, ReAct |

### TIER 16: Audio Processing

Speech and audio applications.

| Notebook | Description |
|----------|-------------|
| [audio-processing](notebooks/audio-processing.ipynb) | FFT, spectrograms, mel, MFCCs |
| [speech-recognition](notebooks/speech-recognition.ipynb) | ASR with CTC loss |

### TIER 17: Generative Models

Learning to generate images and data.

| Notebook | Description |
|----------|-------------|
| [denoising-image](notebooks/denoising-image.ipynb) | Image denoising with conv AEs |
| [vae](notebooks/vae.ipynb) | VAE: reparameterization trick |
| [vqvae](notebooks/vqvae.ipynb) | VQ-VAE: discrete latent spaces |
| [gan-fundamentals](notebooks/gan-fundamentals.ipynb) | GANs: minimax game, mode collapse |
| [generation-image](notebooks/generation-image.ipynb) | VAEs and DCGANs |
| [diffusion-models](notebooks/diffusion-models.ipynb) | DDPM and DDIM sampling |
| [latent-diffusion](notebooks/latent-diffusion.ipynb) | Latent Diffusion and Stable Diffusion |

### TIER 18: Model Optimization & Interpretability

Making models efficient and understandable.

| Notebook | Description |
|----------|-------------|
| [model-interpretability](notebooks/model-interpretability.ipynb) | LIME, SHAP, integrated gradients |
| [knowledge-distillation](notebooks/knowledge-distillation.ipynb) | Teacher-student training |
| [model-pruning](notebooks/model-pruning.ipynb) | Unstructured/structured pruning |
| [1bit-neural-networks](notebooks/1bit-neural-networks.ipynb) | Binary/ternary quantization |

### TIER 19: Advanced Architectures

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

### TIER 20: Deep Reinforcement Learning

Learning through interaction.

| Notebook | Description |
|----------|-------------|
| [rl-policy-gradients](notebooks/rl-policy-gradients.ipynb) | REINFORCE, Actor-Critic, PPO |
| [rl-deep-q-networks](notebooks/rl-deep-q-networks.ipynb) | DQN with experience replay |
| [rl-advanced-policy-methods](notebooks/rl-advanced-policy-methods.ipynb) | A2C/A3C, SAC, TD3 |
| [wip-alphazero](notebooks/wip-alphazero.ipynb) | AlphaZero: MCTS + neural networks |
| [rlhf-alignment](notebooks/rlhf-alignment.ipynb) | RLHF: SFT, reward modeling, PPO, DPO |
| [rl-model-based](notebooks/rl-model-based.ipynb) | World models, Dyna, planning |

### TIER 21: Fascinating Phenomena

Research findings that challenge intuitions.

| Notebook | Description |
|----------|-------------|
| [adversarial-robustness](notebooks/adversarial-robustness.ipynb) | FGSM/PGD attacks, adversarial training |
| 🔴 [double-descent](notebooks/double-descent.ipynb) | Test error decreasing beyond interpolation |
| 🔴 [grokking](notebooks/grokking.ipynb) | Sudden generalization after overfitting |

### TIER 22: Alternative Learning Paradigms

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
uv run python run_sweep.py sweeps/config.yaml notebooks/notebook.ipynb --count 10
```

See `sweeps/*.yaml` for configuration examples (bayes, grid, random).

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

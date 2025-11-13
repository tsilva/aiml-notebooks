# 🧠 aiml-notebooks

<p align="center">
  <img src="logo.jpg" alt="Logo" width="400"/>
</p>

AI/ML Jupyter notebooks for learning and experimentation. This repository contains educational notebooks covering fundamental to advanced AI/ML concepts, organized in optimal learning order to take you from absolute beginner to advanced practitioner.

## 📖 Overview

This repository provides a comprehensive, hands-on learning path through machine learning and deep learning. Each notebook is designed to build deep intuitions through progressive implementation, starting from absolute foundations and building to state-of-the-art techniques.

**The learning path is organized by conceptual prerequisites** - follow the tier progression for the most effective learning experience. Notebooks marked with  are foundational and critical for understanding subsequent material.

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
git clone https://github.com/tsilva/aiml-notebooks.git
cd aiml-notebooks
uv sync
```

### Running Jupyter

```bash
uv run jupyter lab
```

---

## 📚 Learning Path

### 🎯 How to Use This Learning Path

**This learning path is organized by conceptual prerequisites** - each tier builds on previous tiers. The sequence is designed to take a complete AI/ML beginner to advanced mastery by following the specified order.

### Critical Constraint:
When notebooks are added or modified, they must be inserted in their appropriate tier based on conceptual prerequisites. The learning progression from foundational to advanced topics is the PRIMARY organizing principle.

---

## TIER 1: Absolute Foundations

**Start here!** These are the bare essentials before anything else.

###  **[numpy-fundamentals.ipynb](notebooks/numpy-fundamentals.ipynb)**
NumPy arrays, indexing, broadcasting, and operations. Foundation for all numerical ML work.

###  **[linear-algebra.ipynb](notebooks/linear-algebra.ipynb)** 
Vectors, matrices, dot products, norms, eigenvalues, and orthogonality. Mathematical foundation for all ML algorithms.

###  **[tensors-operations.ipynb](notebooks/tensors-operations.ipynb)**
PyTorch tensor operations: shapes, broadcasting, indexing, and basic algebra.

###  **[tensor-multiplication.ipynb](notebooks/tensor-multiplication.ipynb)**
Matrix multiplication and broadcasting in detail. How neural network layers compute.

###  **[calculus-refresher.ipynb](notebooks/calculus-refresher.ipynb)** 
Derivatives, chain rule, partial derivatives, and gradients. Foundation for backpropagation and optimization.

###  **[linear-regression.ipynb](notebooks/linear-regression.ipynb)**
Linear regression from scratch using gradient descent vs closed-form solution.

###  **[metrics.ipynb](notebooks/metrics.ipynb)**
Evaluation metrics: accuracy, precision, recall, F1, ROC-AUC. When to use each.

###  **[probability-distributions.ipynb](notebooks/probability-distributions.ipynb)** 
Discrete (Bernoulli, Categorical, Binomial) and continuous (Uniform, Normal) distributions. Foundation for loss functions and generative models.

###  **[loss-functions.ipynb](notebooks/loss-functions.ipynb)**
MSE, MAE, cross-entropy, and hinge loss. Properties and when to use each.

###  **[activation-functions.ipynb](notebooks/activation-functions.ipynb)**
ReLU, sigmoid, tanh and their effects on gradient flow and training.

###  **[logistic-regression.ipynb](notebooks/logistic-regression.ipynb)**
Binary classification with logistic regression, gradient descent, and decision boundaries.

###  **[softmax-multiclass.ipynb](notebooks/softmax-multiclass.ipynb)**
Softmax function, one-hot encoding, and categorical cross-entropy for multiclass classification.

---

## TIER 2: Core ML Principles

**Understanding how to build models that generalize.**

###  **[overfitting-underfitting.ipynb](notebooks/overfitting-underfitting.ipynb)**
Overfitting vs underfitting, model capacity, learning curves, and the bias-variance tradeoff.

###  **[bias-variance-tradeoff.ipynb](notebooks/bias-variance-tradeoff.ipynb)**
Bias-variance decomposition and how it guides model selection.

###  **[train-test-val-split.ipynb](notebooks/train-test-val-split.ipynb)**
Train/validation/test splits, data leakage prevention, and temporal splits for time-series.

###  **[cross-validation.ipynb](notebooks/cross-validation.ipynb)**
K-Fold, Stratified K-Fold, and Time Series cross-validation techniques.

###  **[regularization.ipynb](notebooks/regularization.ipynb)**
L1, L2, dropout, and early stopping to prevent overfitting.

###  **[confidence-intervals.ipynb](notebooks/confidence-intervals.ipynb)**
Quantifying uncertainty in predictions and metrics using bootstrap and hypothesis testing.

### [x] **[hyperparameter-tuning.ipynb](notebooks/hyperparameter-tuning.ipynb)**
Grid search, random search, and Bayesian optimization for hyperparameter tuning.

###  **[ensemble-methods.ipynb](notebooks/ensemble-methods.ipynb)**
Bagging, Random Forests, boosting (AdaBoost, Gradient Boosting), and stacking.

---

## TIER 3: Data & Optimization

**Making the most of your data and training.**

###  **[data-normalization.ipynb](notebooks/data-normalization.ipynb)** 
Min-max scaling vs standardization. Why normalization matters and avoiding data leakage.

###  **[dimensionality-reduction.ipynb](notebooks/dimensionality-reduction.ipynb)**
PCA, t-SNE, and UMAP for reducing dimensions and visualization.

###  **[feature-engineering.ipynb](notebooks/feature-engineering.ipynb)**
Encoding categorical variables, handling missing values, and creating features.

###  **[data-augmentation.ipynb](notebooks/data-augmentation.ipynb)**
Augmentation techniques for images and text to expand training data.

###  **[imbalanced-data.ipynb](notebooks/imbalanced-data.ipynb)**
Handling class imbalance: proper metrics, class weights, resampling, and SMOTE.

###  **[mini-batch-gradient-descent.ipynb](notebooks/mini-batch-gradient-descent.ipynb)** 
Full-batch, mini-batch, and stochastic gradient descent. Epochs, steps, and batch sizes.

###  **[optimizers.ipynb](notebooks/optimizers.ipynb)**
SGD, Momentum, RMSprop, and Adam optimizers. When to use each.

###  **[learning-rate-schedules.ipynb](notebooks/learning-rate-schedules.ipynb)**
Learning rate schedules: step decay, exponential decay, cosine annealing, and warmup.

---

## TIER 4: Deep Learning Foundations

**Now we can start with neural networks!**

###  **[neural-network-fundamentals.ipynb](notebooks/neural-network-fundamentals.ipynb)**
Neurons, layers, networks, forward pass, and the Universal Approximation Theorem.

###  **[computational-graphs.ipynb](notebooks/computational-graphs.ipynb)** 
Representing computations as DAGs. How graphs encode the chain rule for backpropagation.

###  **[z2h-01-backprop.ipynb](notebooks/z2h-01-backprop.ipynb)** 
Backpropagation from scratch: building a micrograd-style autograd engine.

###  **[pytorch-fundamentals.ipynb](notebooks/pytorch-fundamentals.ipynb)**
PyTorch autograd, nn.Parameter, nn.Module, and the training loop.

###  **[gradient-stability.ipynb](notebooks/gradient-stability.ipynb)**
Vanishing and exploding gradient problems in deep networks.

###  **[gradient-clipping.ipynb](notebooks/gradient-clipping.ipynb)**
Gradient clipping to prevent exploding gradients.

###  **[gradient-normalization.ipynb](notebooks/gradient-normalization.ipynb)**
Gradient normalization for stable training across layers.

###  **[batch-normalization.ipynb](notebooks/batch-normalization.ipynb)**
Batch normalization to stabilize and accelerate training.

###  **[layer-normalization.ipynb](notebooks/layer-normalization.ipynb)**
Layer normalization for RNNs and transformers.

---

## TIER 5: First Neural Networks

**Simple neural networks for language modeling.**

###  **[z2h-02-bigram-lm.ipynb](notebooks/z2h-02-bigram-lm.ipynb)**
Character-level bigram language models using frequency counts and PyTorch.

###  **[z2h-03-mlp-lm.ipynb](notebooks/z2h-03-mlp-lm.ipynb)**
Multi-layer perceptron for character-level language modeling.

###  **[z2h-04-optimization-pt1.ipynb](notebooks/z2h-04-optimization-pt1.ipynb)**
Weight initialization (Xavier, Kaiming) and analyzing activation/gradient flow.

###  **[debugging-neural-networks.ipynb](notebooks/debugging-neural-networks.ipynb)**
Systematic debugging: sanity checks, common failures, gradient flow, and learning curves.

---

## TIER 6: Convolutional Neural Networks

**Computer vision fundamentals.**

###  **[tensors-convolution.ipynb](notebooks/tensors-convolution.ipynb)**
Convolutional operations: filters, padding, stride, dilation, and transposed convolutions.

###  **[pooling-operations.ipynb](notebooks/pooling-operations.ipynb)**
MaxPooling, AveragePooling, and Global pooling for downsampling feature maps.

###  **[classification-image.ipynb](notebooks/classification-image.ipynb)**
Building CNN and MLP image classifiers on CIFAR-10, MNIST, and Fashion-MNIST.

###  **[residual-connections.ipynb](notebooks/residual-connections.ipynb)**  **REVOLUTIONARY**
Skip connections and ResNet blocks. Foundation for modern deep architectures.

###  **[transfer-learning.ipynb](notebooks/transfer-learning.ipynb)** 
Using pretrained models: feature extraction vs fine-tuning.

###  **[knowledge-distillation.ipynb](notebooks/knowledge-distillation.ipynb)** 
Compressing models with teacher-student training and soft targets.

###  **[unet-architecture.ipynb](notebooks/unet-architecture.ipynb)** 
U-Net encoder-decoder with skip connections for semantic segmentation.

###  **[grad-cam-visualization.ipynb](notebooks/grad-cam-visualization.ipynb)** 
Grad-CAM for visualizing which image regions influence CNN predictions.

###  **[object-detection-yolo.ipynb](notebooks/object-detection-yolo.ipynb)** 
YOLO architecture: bounding boxes, anchor boxes, and Non-Maximum Suppression.

---

## TIER 7: Information Theory & Embeddings

**Important concepts for advanced topics.**

###  **[kl-divergence.ipynb](notebooks/kl-divergence.ipynb)**
Information theory from the ground up: surprisal, entropy, cross-entropy, and KL divergence. Builds intuition through analogies and visualizations. Essential for understanding VAEs, RL policy optimization, and model training.

###  **[tokenization-methods.ipynb](notebooks/tokenization-methods.ipynb)**
Character, word, BPE, and WordPiece tokenization. Vocabulary vs sequence length tradeoff.

###  **[embeddings.ipynb](notebooks/embeddings.ipynb)**
Learned vector representations for discrete objects (words, items, users).

###  **[word2vec-from-scratch.ipynb](notebooks/word2vec-from-scratch.ipynb)**
Skip-gram Word2Vec with negative sampling. Semantic relationships via vector arithmetic.

###  **[siamese-networks.ipynb](notebooks/siamese-networks.ipynb)** 
Twin networks with shared weights and contrastive loss for similarity learning.

---

## TIER 8: Unsupervised Learning Basics

**Learning without labels.**

###  **[kmeans-from-scratch.ipynb](notebooks/kmeans-from-scratch.ipynb)**
K-means clustering: iterative assignment, k-means++, and the elbow method.

###  **[gaussian-mixture-models.ipynb](notebooks/gaussian-mixture-models.ipynb)**
Gaussian Mixture Models and Expectation-Maximization for soft clustering.

###  **[autoencoders.ipynb](notebooks/autoencoders.ipynb)** 
Encoder-decoder architecture for compression and reconstruction. Foundation for VAEs.

---

## TIER 9: Recurrent Neural Networks

**Sequential data and memory.**

###  **[rnn-from-scratch.ipynb](notebooks/rnn-from-scratch.ipynb)**
RNN with hidden states and backpropagation through time (BPTT).

###  **[lstm-from-scratch.ipynb](notebooks/lstm-from-scratch.ipynb)**
LSTM gates (forget, input, output) and solving vanishing gradients.

---

## TIER 10: Attention & Transformers

**Modern sequence modeling - the revolution.**

###  **[attention-mechanism.ipynb](notebooks/attention-mechanism.ipynb)**  **FOUNDATIONAL CONCEPT**
Scaled dot-product attention: queries, keys, values, and attention weights.

###  **[seq2seq-with-attention.ipynb](notebooks/seq2seq-with-attention.ipynb)**
Seq2seq encoder-decoder with attention. Solving the bottleneck problem.

###  **[positional-encodings.ipynb](notebooks/positional-encodings.ipynb)** 
Sinusoidal, Learned, Relative Position Bias, RoPE, and ALiBi positional encodings.

###  **[transformer-from-scratch.ipynb](notebooks/transformer-from-scratch.ipynb)** 
Full Transformer: self-attention, multi-head attention, encoder-decoder architecture.

###  **[gpt-architecture.ipynb](notebooks/gpt-architecture.ipynb)** 
GPT decoder-only architecture. Causal masking and autoregressive generation.

###  **[bert-architecture.ipynb](notebooks/bert-architecture.ipynb)** 
BERT encoder architecture. Masked language modeling and bidirectional context.

###  **[vision-transformers.ipynb](notebooks/vision-transformers.ipynb)**
Vision Transformers (ViT): patch embeddings and 2D positional encodings.

###  **[mamba-state-space-models.ipynb](notebooks/mamba-state-space-models.ipynb)** 
State Space Models and Mamba: linear-complexity alternative to Transformers.

###  **[lora-peft.ipynb](notebooks/lora-peft.ipynb)**
LoRA and parameter-efficient fine-tuning. Low-rank weight updates.

---

## TIER 11: Text & Audio Applications

**Practical NLP and audio tasks end-to-end.**

###  **[classification-text.ipynb](notebooks/classification-text.ipynb)**
Sentiment analysis: TF-IDF + Logistic Regression, LSTM, and BERT.

###  **[generation-text.ipynb](notebooks/generation-text.ipynb)**
Text generation from bigrams to transformers. Sampling strategies.

###  **[reconstruction-text.ipynb](notebooks/reconstruction-text.ipynb)**
Seq2seq autoencoders for text reconstruction and latent space interpolation.

###  **[denoising-text.ipynb](notebooks/denoising-text.ipynb)**
Text correction with seq2seq and transformers. CER/WER metrics.

###  **[audio-processing.ipynb](notebooks/audio-processing.ipynb)**
Fourier Transform, spectrograms, mel spectrograms, MFCCs, and audio augmentation.

### **[speech-recognition.ipynb](notebooks/speech-recognition.ipynb)**
ASR with CTC loss and RNN-CTC. CER/WER evaluation.

###  **[ner-token-classification.ipynb](notebooks/ner-token-classification.ipynb)** 
Named Entity Recognition with BIO tagging, BiLSTM-CRF, and BERT.

###  **[qa-extractive.ipynb](notebooks/qa-extractive.ipynb)** 
Extractive QA with BiDAF and BERT. Span prediction for SQuAD-style tasks.

###  **[rag-retrieval-augmented.ipynb](notebooks/rag-retrieval-augmented.ipynb)** 
RAG pipeline: chunking, embeddings, vector search, and prompt augmentation.

###  **[prompt-engineering-llms.ipynb](notebooks/prompt-engineering-llms.ipynb)** 
Zero-shot, few-shot, chain-of-thought, and ReAct prompting techniques.

###  **[time-series-forecasting.ipynb](notebooks/time-series-forecasting.ipynb)** 
Time series with MLP, LSTM, and Transformer. Temporal data splitting.

---

## TIER 12: Generative Models for Images

**Learning to generate and reconstruct images.**

###  **[vae.ipynb](notebooks/vae.ipynb)**
Variational Autoencoders (VAE): probabilistic latent spaces, reparameterization trick, and structured generation. Foundation for modern generative models.

###  **[reconstruction-image.ipynb](notebooks/reconstruction-image.ipynb)**
Image reconstruction with Vanilla AE, VAE, and VQ-VAE on MNIST.

###  **[generation-image.ipynb](notebooks/generation-image.ipynb)**  **VAEs & GANs**
VAEs and DCGANs for image generation. Latent space exploration.

###  **[diffusion-models.ipynb](notebooks/diffusion-models.ipynb)** 
Denoising Diffusion (DDPM): forward/reverse diffusion and DDIM sampling.

###  **[latent-diffusion.ipynb](notebooks/latent-diffusion.ipynb)** 
Latent Diffusion and Stable Diffusion. VAE latents with cross-attention conditioning.

###  **[denoising-image.ipynb](notebooks/denoising-image.ipynb)**
Image denoising with convolutional autoencoders and DAE.

---

## TIER 13: Advanced Architectures

**Beyond standard feedforward, convolutional, and recurrent networks.**

###  **[clip-architecture.ipynb](notebooks/clip-architecture.ipynb)** 
CLIP dual encoders with contrastive learning. Zero-shot classification via text.

###  **[capsule-networks.ipynb](notebooks/capsule-networks.ipynb)**
CapsNets with dynamic routing by agreement. Preserving spatial information.

###  **[mixture-of-experts.ipynb](notebooks/mixture-of-experts.ipynb)** 
Sparse MoE with gating networks. Efficient scaling to trillion-parameter models.

###  **[graph-neural-networks.ipynb](notebooks/graph-neural-networks.ipynb)**
GNNs and message passing for graph-structured data.

###  **[neural-odes.ipynb](notebooks/neural-odes.ipynb)**
Neural ODEs: continuous-depth networks as differential equations.

###  **[energy-based-neural-networks.ipynb](notebooks/energy-based-neural-networks.ipynb)**
Hopfield Networks and RBMs. Energy functions and contrastive divergence.

---

## TIER 14: Fascinating Phenomena

**Interesting research findings that challenge intuitions.**

###  **[adversarial-robustness.ipynb](notebooks/adversarial-robustness.ipynb)** 
Adversarial examples, FGSM/PGD attacks, and adversarial training.

###  **[double-descent.ipynb](notebooks/double-descent.ipynb)**
Double descent: test error decreasing beyond interpolation threshold.

###  **[grokking.ipynb](notebooks/grokking.ipynb)**
Grokking: sudden generalization after prolonged overfitting on algorithmic tasks.

---

## TIER 15: Alternative Training Paradigms

**Beyond standard supervised learning - self-supervised, meta-learning, and alternative training methods.**

###  **[meta-learning-few-shot.ipynb](notebooks/meta-learning-few-shot.ipynb)** 
MAML for few-shot learning. Learning to learn with inner/outer loops.

###  **[contrastive-learning.ipynb](notebooks/contrastive-learning.ipynb)** 
Contrastive learning with InfoNCE loss. SimCLR-style self-supervised learning.

###  **[self-supervised-learning.ipynb](notebooks/self-supervised-learning.ipynb)** 
Self-supervised approaches: Rotation, Jigsaw, Masked Autoencoding, SimCLR, BERT masking.

###  **[forward-forward.ipynb](notebooks/forward-forward.ipynb)**
Forward-Forward algorithm: layer-local learning without backpropagation.

###  **[neuroevolution-tic-tac-toe.ipynb](notebooks/neuroevolution-tic-tac-toe.ipynb)**
Neuroevolution with genetic algorithms for tic-tac-toe.

###  **[1bit-neural-networks.ipynb](notebooks/1bit-neural-networks.ipynb)**
Binary and ternary quantization with straight-through estimator.

###  **[multi-task-learning.ipynb](notebooks/multi-task-learning.ipynb)** 
MTL with hard/soft parameter sharing. Loss balancing strategies.

###  **[curriculum-learning.ipynb](notebooks/curriculum-learning.ipynb)** 
Training from easy to hard. Difficulty metrics and curriculum strategies.

###  **[active-learning.ipynb](notebooks/active-learning.ipynb)** 
Query strategies for selecting data to label: uncertainty, QBC, diversity.

###  **[continual-learning.ipynb](notebooks/continual-learning.ipynb)** 
Lifelong learning without catastrophic forgetting. Replay, EWC, and architecture methods.

---

## TIER 16: Reinforcement Learning

**Learning through interaction with environments.**

###  **[rl-q-learning.ipynb](notebooks/rl-q-learning.ipynb)**  **FOUNDATIONAL RL**
Q-learning, MDPs, Bellman equation, and ε-greedy exploration.

###  **[mcts-simple.ipynb](notebooks/mcts-simple.ipynb)**
Monte Carlo Tree Search with UCB1 for tic-tac-toe.

###  **[rl-policy-gradients.ipynb](notebooks/rl-policy-gradients.ipynb)** 
Policy gradients, REINFORCE, Actor-Critic, and PPO basics.

###  **[rl-deep-q-networks.ipynb](notebooks/rl-deep-q-networks.ipynb)** 
DQN with experience replay and target networks.

###  **[rl-advanced-policy-methods.ipynb](notebooks/rl-advanced-policy-methods.ipynb)** 
A2C/A3C, SAC, and TD3 for continuous control.

###  **[rlhf-alignment.ipynb](notebooks/rlhf-alignment.ipynb)** 
RLHF pipeline: SFT, reward modeling, and PPO optimization. DPO alternative.

###  **[rl-model-based.ipynb](notebooks/rl-model-based.ipynb)**
World models, Dyna algorithm, and planning for sample efficiency.

---

## TIER 17: Fun & Specialized Applications

**Unique and interesting applications.**

###  **[chip8-emulator.ipynb](notebooks/chip8-emulator.ipynb)**
CHIP-8 emulator: fetch-decode-execute cycle and opcodes.

---

## 🗄️ Deprecated / Work in Progress

**Incomplete notebooks not recommended for learning.**

###  **[z2h-05-optimization-pt2-wip.ipynb](notebooks/deprecated/z2h-05-optimization-pt2-wip.ipynb)**
PyTorch experiments with initialization and batch normalization.

###  **[z2h-06-backprop-ninja-wip.ipynb](notebooks/deprecated/z2h-06-backprop-ninja-wip.ipynb)**
Manually implementing backward passes for cross-entropy and batch normalization.

###  **[z2h-07-wavenet-lm-wip.ipynb](notebooks/deprecated/z2h-07-wavenet-lm-wip.ipynb)**
WaveNet-inspired hierarchical language model.

###  **[wip-bit-parity-rnn.ipynb](notebooks/deprecated/wip-bit-parity-rnn.ipynb)**
RNN for bit-parity classification task.

###  **[wip-bit-parity-gru.ipynb](notebooks/deprecated/wip-bit-parity-gru.ipynb)**
GRU with update and reset gates for bit-parity classification.

###  **[wip-rl-world-model-01-repr.ipynb](notebooks/deprecated/wip-rl-world-model-01-repr.ipynb)**
Compressing gameplay frames with convolutional autoencoders.

###  **[wip-rl-world-model-02-dynamics.ipynb](notebooks/deprecated/wip-rl-world-model-02-dynamics.ipynb)**
Learning dynamics models to predict future latent states.

###  **[wip-rl-atari-pong-imitation.ipynb](notebooks/deprecated/wip-rl-atari-pong-imitation.ipynb)**
Imitation learning for Atari Pong.

###  **[wip-debate-generator.ipynb](notebooks/deprecated/wip-debate-generator.ipynb)**
AI debate simulator with LangChain and multi-agent LLMs.

---

## 🎓 Learning Path Recommendations

### **Quick Start Path (Core Essentials)**
If you're short on time, focus on notebooks marked with ⭐:
1. Complete all of Tier 1-4 (foundations)
2. z2h-01-backprop.ipynb
3. kl-divergence.ipynb (for VAEs/RL)
4. transformer-from-scratch.ipynb (modern AI foundation)
5. generation-image.ipynb (VAEs & GANs)

### **Full Path (N00b → God-Tier)**
Follow the tiers in order from 1 → 17 for comprehensive mastery.

### **Specialization Paths**

**Computer Vision Track:**
- Tiers 1-4 → Tier 6 → Tier 12 → Tier 13 (GNNs) → Tier 14

**NLP/LLM Track:**
- Tiers 1-5 → Tier 7 → Tier 9 → Tier 10 → Tier 11 → Tier 14

**Reinforcement Learning Track:**
- Tiers 1-4 → Tier 7 (esp. KL divergence) → Tier 16 (focus on rl-deep-q-networks.ipynb)

**Research/Theory Track:**
- Tiers 1-4 → Tier 7 → Tier 13 → Tier 14 → Tier 15

---

## 🔧 Advanced Usage

### Shared Utilities

The repository includes shared utilities in `src/aiml_notebooks/` for common tasks like data loading, tokenization, and device management. Check the source files for the current API.

### Running Hyperparameter Sweeps

```bash
uv run python run_sweep.py sweeps/config.yaml notebooks/notebook.ipynb --count 10
# See sweeps/*.yaml for config examples (bayes, grid, random)
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

**Note:** This learning progression builds conceptual dependencies systematically. Each tier assumes mastery of previous tiers. Notebooks marked as "Work in Progress" may be incomplete but are placed where they fit conceptually in the learning path.

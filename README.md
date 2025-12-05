# 🧠 aiml-notebooks

<p align="center">
  <img src="logo.jpg" alt="Logo" width="400"/>
</p>

AI/ML Jupyter notebooks for learning and experimentation. This repository contains educational notebooks covering fundamental to advanced AI/ML concepts, organized in optimal learning order to take you from absolute beginner to advanced practitioner.

## 📖 Overview

This repository provides a comprehensive, hands-on learning path through machine learning and deep learning. Each notebook is designed to build deep intuitions through progressive implementation, starting from absolute foundations and building to state-of-the-art techniques.

**The learning path is organized by conceptual prerequisites** - follow the tier progression for the most effective learning experience.

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

**Legend:**
- ✅ = Completed notebook
- 🔴 = Advanced/challenging material
- ⏱️ = Estimated time for tier

**Naming conventions:**
- `z2h-XX-*` notebooks follow Andrej Karpathy's "Zero to Hero" curriculum style

### Critical Constraint:
When notebooks are added or modified, they must be inserted in their appropriate tier based on conceptual prerequisites. The learning progression from foundational to advanced topics is the PRIMARY organizing principle.

---

## 🎯 Learning Tracks

Choose a focused path based on your goals:

| Track | Tiers | Focus |
|-------|-------|-------|
| **🏃 Fast Track to LLMs** | 1 → 2 → 3 → 6 → 7 → 8 → 11 → 12 → 13 | Quickest path to understanding modern language models |
| **👁️ Computer Vision** | 1 → 2 → 3 → 6 → 7 → 9 → 10 → 15 | CNNs, architectures, and image generation |
| **📊 Classical ML Mastery** | 1 → 2 → 3 → 4 → 5 → 6 | Strong foundations before deep learning |
| **🎮 Reinforcement Learning** | 1 → 2 → 3 → 4 → 5A → 6 → 7 → 19 | Decision-making and game-playing agents |
| **🔬 Complete Curriculum** | All tiers in order | Comprehensive understanding of the field |

---

## TIER 1: Numerical Computing ⏱️ ~4-6 hours

**Start here!** Learn to work with arrays and tensors - the data structures of ML.

### ✅ **[numpy-fundamentals.ipynb](notebooks/numpy-fundamentals.ipynb)**
NumPy arrays, indexing, broadcasting, and operations. Foundation for all numerical ML work.

### **[tensors-operations.ipynb](notebooks/tensors-operations.ipynb)**
PyTorch tensor operations: shapes, broadcasting, indexing, and basic algebra.

### **[tensor-multiplication.ipynb](notebooks/tensor-multiplication.ipynb)**
Matrix multiplication and broadcasting in detail. How neural network layers compute.

---

## TIER 2: Mathematical Foundations ⏱️ ~8-12 hours

**The math you need.** Linear algebra, calculus, and probability for ML.

### **[linear-algebra.ipynb](notebooks/linear-algebra.ipynb)**
Vectors, matrices, dot products, norms, eigenvalues, and orthogonality. Mathematical foundation for all ML algorithms.

### **[calculus-refresher.ipynb](notebooks/calculus-refresher.ipynb)**
Derivatives, chain rule, partial derivatives, and gradients. Foundation for backpropagation and optimization.

### **[probability-distributions.ipynb](notebooks/probability-distributions.ipynb)**
Discrete (Bernoulli, Categorical, Binomial) and continuous (Uniform, Normal) distributions. Foundation for loss functions and generative models.

### **[bayesian-ml-basics.ipynb](notebooks/bayesian-ml-basics.ipynb)**
Bayesian machine learning fundamentals: Bayes' theorem, priors (informative, non-informative, conjugate), posteriors, credible intervals, and uncertainty quantification. Implements Bayesian linear regression from scratch. Foundation for understanding regularization as prior beliefs and uncertainty in predictions.

---

## TIER 3: Core ML Components ⏱️ ~6-8 hours

**Essential building blocks.** Components used in every ML model.

### **[data-normalization.ipynb](notebooks/data-normalization.ipynb)**
Min-max scaling vs standardization. Why normalization matters and avoiding data leakage.

### **[loss-functions.ipynb](notebooks/loss-functions.ipynb)**
MSE, MAE, cross-entropy, and hinge loss. Properties and when to use each.

### **[activation-functions.ipynb](notebooks/activation-functions.ipynb)**
ReLU, sigmoid, tanh and their effects on gradient flow and training.

### **[metrics.ipynb](notebooks/metrics.ipynb)**
Evaluation metrics: accuracy, precision, recall, F1, ROC-AUC. When to use each.

### **[kl-divergence.ipynb](notebooks/kl-divergence.ipynb)**
Information theory from the ground up: surprisal, entropy, cross-entropy, and KL divergence. Builds intuition through analogies and visualizations. Essential for understanding VAEs, RL policy optimization, and model training.

---

## TIER 4: Classical ML Algorithms ⏱️ ~10-14 hours

**Traditional machine learning.** Master these before deep learning.

### **[linear-regression.ipynb](notebooks/linear-regression.ipynb)**
Linear regression from scratch using gradient descent vs closed-form solution.

### **[logistic-regression.ipynb](notebooks/logistic-regression.ipynb)**
Binary classification with logistic regression, gradient descent, and decision boundaries.

### **[softmax-multiclass.ipynb](notebooks/softmax-multiclass.ipynb)**
Softmax function, one-hot encoding, and categorical cross-entropy for multiclass classification.

### **[decision-trees.ipynb](notebooks/decision-trees.ipynb)**
Decision trees from scratch: recursive splitting, impurity measures (Gini, Entropy), information gain, regularization, and feature importance. Foundation for ensemble methods (Random Forests, Gradient Boosting).

### **[naive-bayes.ipynb](notebooks/naive-bayes.ipynb)**
Naive Bayes classifiers: Bayes' theorem for classification, the naive independence assumption, Gaussian NB (continuous features), Multinomial NB (text/counts), Bernoulli NB (binary features), and Laplace smoothing. Fast, simple, and surprisingly effective for text classification and as a baseline.

### **[support-vector-machines.ipynb](notebooks/support-vector-machines.ipynb)**
Support Vector Machines (SVMs): maximum margin classification, support vectors, the kernel trick (linear, polynomial, RBF), C and gamma hyperparameters, and when to use SVMs vs other classifiers. A classic algorithm with elegant mathematical foundations worth understanding.

### **[kmeans-from-scratch.ipynb](notebooks/kmeans-from-scratch.ipynb)**
K-means clustering: iterative assignment, k-means++, and the elbow method.

### **[gaussian-mixture-models.ipynb](notebooks/gaussian-mixture-models.ipynb)**
Gaussian Mixture Models and Expectation-Maximization for soft clustering.

---

## TIER 5: Generalization & Model Selection ⏱️ ~8-10 hours

**Understanding generalization.** How to build models that work on new data.

### **[bias-variance-overfitting.ipynb](notebooks/bias-variance-overfitting.ipynb)**
Bias-variance tradeoff, overfitting vs underfitting, model capacity, learning curves, and practical diagnosis. Comprehensive coverage of how model complexity affects generalization.

### **[train-test-val-split.ipynb](notebooks/train-test-val-split.ipynb)**
Train/validation/test splits, data leakage prevention, and temporal splits for time-series.

### **[cross-validation.ipynb](notebooks/cross-validation.ipynb)**
K-Fold, Stratified K-Fold, and Time Series cross-validation techniques.

### **[regularization.ipynb](notebooks/regularization.ipynb)**
L1, L2, dropout, and early stopping to prevent overfitting.

### **[confidence-intervals.ipynb](notebooks/confidence-intervals.ipynb)**
Quantifying uncertainty in predictions and metrics using bootstrap and hypothesis testing.

### **[hyperparameter-tuning.ipynb](notebooks/hyperparameter-tuning.ipynb)**
Grid search, random search, and Bayesian optimization for hyperparameter tuning.

### **[ensemble-methods.ipynb](notebooks/ensemble-methods.ipynb)**
Bagging, Random Forests, boosting (AdaBoost, Gradient Boosting), and stacking.

---

## TIER 5A: RL Foundations ⏱️ ~4-6 hours

**Decision-making basics.** Core RL concepts that don't require deep learning.

### **[rl-q-learning.ipynb](notebooks/rl-q-learning.ipynb)**
Q-learning, MDPs, Bellman equation, and ε-greedy exploration.

### **[mcts-simple.ipynb](notebooks/mcts-simple.ipynb)**
Monte Carlo Tree Search with UCB1 for tic-tac-toe.

---

## TIER 6: Data Engineering ⏱️ ~6-8 hours

**Making the most of your data.**

### **[dimensionality-reduction.ipynb](notebooks/dimensionality-reduction.ipynb)**
PCA, t-SNE, and UMAP for reducing dimensions and visualization.

### **[feature-engineering.ipynb](notebooks/feature-engineering.ipynb)**
Encoding categorical variables, handling missing values, and creating features.

### **[data-augmentation.ipynb](notebooks/data-augmentation.ipynb)**
Augmentation techniques for images and text to expand training data.

### **[imbalanced-data.ipynb](notebooks/imbalanced-data.ipynb)**
Handling class imbalance: proper metrics, class weights, resampling, and SMOTE.

---

## TIER 7: Deep Learning Foundations ⏱️ ~12-16 hours

**Neural networks from scratch.** The core of modern ML.

### Hardware Foundations

### **[gpu-hardware-basics.ipynb](notebooks/gpu-hardware-basics.ipynb)**
GPU and hardware fundamentals: understanding CUDA cores, memory hierarchy (registers, shared memory, L1/L2 cache, global memory), memory bandwidth bottlenecks, coalesced vs strided access patterns, arithmetic intensity, Tensor Cores, and practical optimization guidelines. Essential for understanding why certain operations are fast/slow, why batch size matters, and how to write GPU-friendly code. Read this before diving into training and optimization to build intuition about hardware constraints.

### Core Concepts

### **[neural-network-fundamentals.ipynb](notebooks/neural-network-fundamentals.ipynb)**
Neurons, layers, networks, forward pass, and the Universal Approximation Theorem.

### **[computational-graphs.ipynb](notebooks/computational-graphs.ipynb)**
Representing computations as DAGs. How graphs encode the chain rule for backpropagation.

### **[z2h-01-backprop.ipynb](notebooks/z2h-01-backprop.ipynb)**
Backpropagation from scratch: building a micrograd-style autograd engine.

### **[pytorch-fundamentals.ipynb](notebooks/pytorch-fundamentals.ipynb)**
PyTorch autograd, nn.Parameter, nn.Module, and the training loop.

### Optimization

### ✅ **[mini-batch-gradient-descent.ipynb](notebooks/mini-batch-gradient-descent.ipynb)**
Full-batch, mini-batch, and stochastic gradient descent. Epochs, steps, and batch sizes.

### ✅ **[optimizers.ipynb](notebooks/optimizers.ipynb)**
SGD, Momentum, RMSprop, and Adam optimizers. When to use each.

### ✅ **[learning-rate-schedules.ipynb](notebooks/learning-rate-schedules.ipynb)**
Learning rate schedules: step decay, exponential decay, cosine annealing, and warmup.

### Gradient Flow & Stability

### ✅ **[gradient-stability.ipynb](notebooks/gradient-stability.ipynb)**
Vanishing and exploding gradient problems in deep networks.

### **[gradient-stabilization.ipynb](notebooks/gradient-stabilization.ipynb)**
Gradient clipping and normalization for stable training: value vs norm clipping, geometric intuition, threshold selection, and when to use each technique.

### **[batch-normalization.ipynb](notebooks/batch-normalization.ipynb)**
Batch normalization to stabilize and accelerate training.

### ✅ **[layer-normalization.ipynb](notebooks/layer-normalization.ipynb)**
Layer normalization for RNNs and transformers.

### **[z2h-04-optimization-pt1.ipynb](notebooks/z2h-04-optimization-pt1.ipynb)**
Weight initialization (Xavier, Kaiming) and analyzing activation/gradient flow.

### **[debugging-neural-networks.ipynb](notebooks/debugging-neural-networks.ipynb)**
Systematic debugging: sanity checks, common failures, gradient flow, and learning curves.

---

## TIER 8: Text & Embedding Foundations ⏱️ ~6-8 hours

**Essential text processing.** Required before language models.

### **[tokenization-methods.ipynb](notebooks/tokenization-methods.ipynb)**
Character, word, BPE, and WordPiece tokenization. Vocabulary vs sequence length tradeoff.

### **[embeddings.ipynb](notebooks/embeddings.ipynb)**
Learned vector representations for discrete objects (words, items, users).

### **[word2vec-from-scratch.ipynb](notebooks/word2vec-from-scratch.ipynb)**
Skip-gram Word2Vec with negative sampling. Semantic relationships via vector arithmetic.

---

## TIER 9: First Neural Networks ⏱️ ~6-8 hours

**Simple neural networks for language modeling.**

### **[z2h-02-bigram-lm.ipynb](notebooks/z2h-02-bigram-lm.ipynb)**
Character-level bigram language models using frequency counts and PyTorch.

### **[z2h-03-mlp-lm.ipynb](notebooks/z2h-03-mlp-lm.ipynb)**
Multi-layer perceptron for character-level language modeling.

---

## TIER 10: Convolutional Neural Networks ⏱️ ~16-24 hours

**Computer vision fundamentals.**

### Fundamentals

### **[tensors-convolution.ipynb](notebooks/tensors-convolution.ipynb)**
Convolutional operations: filters, padding, stride, dilation, and transposed convolutions.

### **[pooling-operations.ipynb](notebooks/pooling-operations.ipynb)**
MaxPooling, AveragePooling, and Global pooling for downsampling feature maps.

### **[cnn-architectures-evolution.ipynb](notebooks/cnn-architectures-evolution.ipynb)**
Historical evolution of CNNs from LeNet (1998) → AlexNet (2012) → VGG (2014) → Inception (2014). Shows WHY each innovation mattered: ReLU activation, dropout regularization, depth with small filters, and multi-scale features. Read this overview first, then dive into individual architectures below.

### Architecture Deep Dives (Optional but Recommended)

*Follow the evolution to understand WHY each innovation happened:*

### **[lenet-limitations.ipynb](notebooks/lenet-limitations.ipynb)**
LeNet-5 (1998): the first successful CNN architecture. Implementation and training on CIFAR-10 reveals fundamental limitations: insufficient depth (only 2 conv layers), minimal feature capacity (6/16 filters), no regularization, and FC layers that don't scale. Each limitation motivates specific innovations in later architectures. Essential foundation for understanding CNN evolution.

### **[alexnet-breakthrough.ipynb](notebooks/alexnet-breakthrough.ipynb)**
AlexNet (2012): the ImageNet breakthrough that started the deep learning revolution. Shows how it solves every LeNet limitation through ReLU activation, dropout, depth (5 conv layers), massive capacity (64→192→384 filters), and data augmentation. Then reveals AlexNet's own inefficiencies: large filters waste parameters, heterogeneous design is hard to scale, and FC layers still dominate. Essential for understanding why VGG chose uniform 3×3 convolutions.

### **[vgg-depth-uniformity.ipynb](notebooks/vgg-depth-uniformity.ipynb)**
VGG (2014): proves depth matters more than filter size through uniform 3×3 architecture. Shows mathematically that stacked 3×3 filters achieve same receptive field as large filters but with fewer parameters and more non-linearity. Demonstrates elegant systematic depth scaling (VGG-11/13/16/19). Then reveals VGG's limitations: massive parameter count (FC layers dominate), single-scale features (no multi-scale extraction), and high memory/compute costs. Motivates Inception's multi-scale parallel approach.

### **[inception-multiscale.ipynb](notebooks/inception-multiscale.ipynb)**
Inception/GoogLeNet (2014): answers "what filter size?" with "all of them!" Shows multi-scale parallel architecture (1×1, 3×3, 5×5, pool in parallel), 1×1 bottleneck layers for 97% parameter reduction, and Global Average Pooling to eliminate FC layers entirely. Achieves better performance than VGG with 27× fewer parameters (5M vs 138M). Then reveals the final challenge: degradation problem preventing networks beyond ~20 layers. Sets up the need for ResNet's skip connections to enable 100+ layer networks.

### **[resnet-skip-connections.ipynb](notebooks/resnet-skip-connections.ipynb)**
ResNet (2015): solves the degradation problem that prevented networks beyond ~20 layers. Explains mathematically why learning residuals F(x) is easier than learning direct mappings H(x), and how skip connections create gradient highways. Implements ResNet-18 with BasicBlocks, demonstrates successful training of 18+ layer networks, and shows why ResNet became the foundation for virtually all modern architectures (Transformers, U-Net, DenseNet). First architecture to exceed human-level performance on ImageNet (3.57% vs 5% error). Completes the CNN evolution story from LeNet to mature deep learning.

### ✅ **[why-residual-connections-work.ipynb](notebooks/why-residual-connections-work.ipynb)**
Deep dive into why skip connections work: gradient highways preventing vanishing gradients, ResNets as implicit ensembles of 2^n paths, and feature reuse across layers.

### Efficient Architectures

### **[mobilenet-efficient-cnns.ipynb](notebooks/mobilenet-efficient-cnns.ipynb)**
MobileNet (2017): shifts focus from pure accuracy to efficiency for mobile and edge devices. Introduces depthwise separable convolutions that factorize standard convolutions into depthwise (spatial filtering per channel) and pointwise (channel mixing) operations, achieving 8-9× speedup with minimal accuracy loss. Implements width multiplier for easy model scaling (0.25×, 0.5×, 0.75×, 1.0×). Shows the efficiency vs accuracy trade-off landscape and demonstrates that architectural innovations can reduce computational cost while maintaining performance. Opens the efficiency branch of CNN evolution (MobileNet → EfficientNet → NAS).

### 🔴 **[efficientnet-compound-scaling.ipynb](notebooks/efficientnet-compound-scaling.ipynb)**
EfficientNet (2019): solves the scaling problem through compound scaling that simultaneously balances depth, width, and resolution with fixed ratios (d=α^φ, w=β^φ, r=γ^φ). Shows why single-dimension scaling is suboptimal and uses Neural Architecture Search (NAS) to discover optimal base architecture. Implements MBConv blocks with Squeeze-and-Excitation for channel attention. Achieves 10× better efficiency than previous CNNs - EfficientNet-B0 matches ResNet-50 accuracy with 5× fewer parameters. Completes the efficiency branch: shows how automated search + systematic scaling creates the peak of CNN efficiency before Vision Transformers.

### Applications & Techniques

### **[classification-image.ipynb](notebooks/classification-image.ipynb)**
Building CNN and MLP image classifiers on CIFAR-10, MNIST, and Fashion-MNIST.

### **[transfer-learning.ipynb](notebooks/transfer-learning.ipynb)**
Using pretrained models: feature extraction vs fine-tuning.

### **[neural-style-transfer.ipynb](notebooks/neural-style-transfer.ipynb)**
Neural Style Transfer: artistic image generation with CNNs. Uses pretrained VGG-19 to extract content and style features, implements Gram matrices for capturing artistic style, and optimizes pixel space to blend content with artistic style. Shows how CNNs naturally separate content from style in their representations.

### **[unet-architecture.ipynb](notebooks/unet-architecture.ipynb)**
U-Net encoder-decoder with skip connections for semantic segmentation.

### **[grad-cam-visualization.ipynb](notebooks/grad-cam-visualization.ipynb)**
Grad-CAM for visualizing which image regions influence CNN predictions.

### **[object-detection-yolo.ipynb](notebooks/object-detection-yolo.ipynb)**
YOLO architecture: bounding boxes, anchor boxes, and Non-Maximum Suppression.

---

## TIER 11: Recurrent Neural Networks ⏱️ ~8-10 hours

**Sequential data and memory.**

### **[rnn-from-scratch.ipynb](notebooks/rnn-from-scratch.ipynb)**
RNN with hidden states and backpropagation through time (BPTT).

### **[lstm-from-scratch.ipynb](notebooks/lstm-from-scratch.ipynb)**
LSTM gates (forget, input, output) and solving vanishing gradients.

### **[gru-from-scratch.ipynb](notebooks/gru-from-scratch.ipynb)**
GRU: the simpler gated architecture with 2 gates instead of 3. Update gate (combines forget+input), reset gate, gradient highways, and when to choose GRU vs LSTM.

### **[time-series-forecasting.ipynb](notebooks/time-series-forecasting.ipynb)**
Time series with MLP, LSTM, and Transformer. Temporal data splitting and walk-forward validation.

---

## TIER 12: Attention & Transformers ⏱️ ~12-16 hours

**Modern sequence modeling - the revolution.**

*Prerequisites: TIER 7 (backprop, PyTorch), TIER 8 (embeddings), TIER 11 (RNNs for context)*

### Foundations

### ✅ **[attention-mechanism.ipynb](notebooks/attention-mechanism.ipynb)**
Scaled dot-product attention: queries, keys, values, and attention weights.

### **[seq2seq-with-attention.ipynb](notebooks/seq2seq-with-attention.ipynb)**
Seq2seq encoder-decoder with attention. Solving the bottleneck problem.

### **[positional-encodings.ipynb](notebooks/positional-encodings.ipynb)**
Sinusoidal, Learned, Relative Position Bias, RoPE, and ALiBi positional encodings.

### **[transformer-from-scratch.ipynb](notebooks/transformer-from-scratch.ipynb)**
Full Transformer: self-attention, multi-head attention, encoder-decoder architecture.

### Architectures

### ✅ **[gpt-architecture.ipynb](notebooks/gpt-architecture.ipynb)**
GPT decoder-only architecture. Causal masking and autoregressive generation.

### ✅ **[bert-architecture.ipynb](notebooks/bert-architecture.ipynb)**
BERT encoder architecture. Masked language modeling and bidirectional context.

### GPT-2 Implementations

*Three implementations for different learning goals:*

### **[gpt2-from-scratch.ipynb](notebooks/gpt2-from-scratch.ipynb)** ← Start here (educational)
Building GPT-2 from scratch in raw PyTorch. Character-level language model with transformer decoder blocks, training on Shakespeare/LoTR text. Includes tiktoken tokenizer integration and experimental configs.

### **[gpt2-keras-jax.ipynb](notebooks/gpt2-keras-jax.ipynb)** ← Alternative framework
GPT-2 style transformer built from scratch using Keras with JAX backend. Educational implementation showing token/position embeddings, self-attention, multi-head attention, and transformer blocks with residual connections. Demonstrates JAX's XLA compilation benefits.

### Beyond Text

### **[vision-transformers.ipynb](notebooks/vision-transformers.ipynb)**
Vision Transformers (ViT): patch embeddings and 2D positional encodings.

### **[understanding-mamba.ipynb](notebooks/understanding-mamba.ipynb)**
Mamba intuition for Transformer experts: builds from the O(L²) attention problem to selective state spaces using simple examples and visualizations. Shows the ONE key idea (input-dependent recurrence) without control theory. Perfect first read before diving into the mathematical details. Compares directly to LSTMs and attention throughout.

### 🔴 **[mamba-state-space-models.ipynb](notebooks/mamba-state-space-models.ipynb)**
Mamba deep dive: continuous/discrete state space models, HiPPO initialization, structured matrices, and the full mathematical formulation. Comprehensive coverage from control theory foundations to implementation. Read `understanding-mamba.ipynb` first for intuition.

---

## TIER 13: Efficient Transformers ⏱️ ~8-12 hours

**Making transformers fast and deployable.**

### 🔴 **[flash-attention.ipynb](notebooks/flash-attention.ipynb)**
Flash Attention: memory-efficient attention with tiling and online softmax. Explains why standard attention is I/O bound (not compute bound), GPU memory hierarchy (SRAM vs HBM), and implements simplified Flash Attention from scratch. Covers causal masking, memory comparison, and PyTorch's scaled_dot_product_attention.

### **[kv-caching.ipynb](notebooks/kv-caching.ipynb)**
KV Caching: efficient autoregressive inference by caching key-value tensors. Explains why naive generation recomputes attention redundantly, implements KV caching from scratch with prefill/decode phases, analyzes memory requirements for different model sizes, and covers Multi-Query Attention (MQA) and Grouped-Query Attention (GQA) for reducing cache memory.

### **[speculative-decoding.ipynb](notebooks/speculative-decoding.ipynb)**
Speculative Decoding: accelerating LLM inference using a small draft model to propose multiple tokens that are verified in parallel by the target model. Explains why autoregressive generation is memory-bound, implements speculative decoding from scratch with rejection sampling, analyzes acceptance rates and expected speedup, and covers draft model selection strategies (smaller models, distillation, early exit, quantization).

### **[int8-fp16-quantization.ipynb](notebooks/int8-fp16-quantization.ipynb)**
INT8/FP16 Quantization: reducing model precision for efficient deployment. Covers numerical representations (FP32, FP16, BF16, INT8), symmetric and asymmetric quantization, per-tensor vs per-channel granularity, Post-Training Quantization (PTQ) with dynamic and static approaches, Quantization-Aware Training (QAT), and FP16 mixed precision training with AMP. Implements quantization from scratch and with PyTorch's quantization toolkit.

### **[lora-peft.ipynb](notebooks/lora-peft.ipynb)**
LoRA and parameter-efficient fine-tuning. Low-rank weight updates.

---

## TIER 14: Representation Learning ⏱️ ~8-10 hours

**Learning without labels and learning useful representations.**

### **[autoencoders.ipynb](notebooks/autoencoders.ipynb)**
Encoder-decoder architecture for compression and reconstruction. Foundation for VAEs.

### **[siamese-networks.ipynb](notebooks/siamese-networks.ipynb)**
Twin networks with shared weights and contrastive loss for similarity learning.

### **[contrastive-learning.ipynb](notebooks/contrastive-learning.ipynb)**
Contrastive learning with InfoNCE loss. SimCLR-style self-supervised learning.

### **[self-supervised-learning.ipynb](notebooks/self-supervised-learning.ipynb)**
Self-supervised approaches: Rotation, Jigsaw, Masked Autoencoding, SimCLR, BERT masking.

---

## TIER 15: NLP Applications ⏱️ ~10-14 hours

**Practical NLP tasks end-to-end.**

### **[classification-text.ipynb](notebooks/classification-text.ipynb)**
Sentiment analysis: TF-IDF + Logistic Regression, LSTM, and BERT.

### **[generation-text.ipynb](notebooks/generation-text.ipynb)**
Text generation from bigrams to transformers. Sampling strategies.

### **[reconstruction-text.ipynb](notebooks/reconstruction-text.ipynb)**
Seq2seq autoencoders for text reconstruction and latent space interpolation.

### **[denoising-text.ipynb](notebooks/denoising-text.ipynb)**
Text correction with seq2seq and transformers. CER/WER metrics.

### **[ner-token-classification.ipynb](notebooks/ner-token-classification.ipynb)**
Named Entity Recognition with BIO tagging, BiLSTM-CRF, and BERT.

### **[qa-extractive.ipynb](notebooks/qa-extractive.ipynb)**
Extractive QA with BiDAF and BERT. Span prediction for SQuAD-style tasks.

### **[rag-retrieval-augmented.ipynb](notebooks/rag-retrieval-augmented.ipynb)**
RAG pipeline: chunking, embeddings, vector search, and prompt augmentation.

### **[prompt-engineering-llms.ipynb](notebooks/prompt-engineering-llms.ipynb)**
Zero-shot, few-shot, chain-of-thought, and ReAct prompting techniques.

---

## TIER 16: Audio Processing ⏱️ ~4-6 hours

**Speech and audio applications.**

### **[audio-processing.ipynb](notebooks/audio-processing.ipynb)**
Fourier Transform, spectrograms, mel spectrograms, MFCCs, and audio augmentation.

### **[speech-recognition.ipynb](notebooks/speech-recognition.ipynb)**
ASR with CTC loss and RNN-CTC. CER/WER evaluation.

---

## TIER 17: Generative Models ⏱️ ~12-16 hours

**Learning to generate images and other data.**

### **[denoising-image.ipynb](notebooks/denoising-image.ipynb)**
Image denoising with convolutional autoencoders and Denoising Autoencoders (DAE). Simpler foundation before probabilistic models.

### **[vae.ipynb](notebooks/vae.ipynb)**
Variational Autoencoders (VAE): probabilistic latent spaces, reparameterization trick, and structured generation. Foundation for modern generative models.

### **[vqvae.ipynb](notebooks/vqvae.ipynb)**
Vector-Quantized VAE (VQ-VAE): discrete latent spaces, learned codebook, vector quantization, and straight-through estimator. Foundation for DALL-E and hierarchical generation.

### **[gan-fundamentals.ipynb](notebooks/gan-fundamentals.ipynb)**
Deep dive into GANs: the minimax game formulation, generator/discriminator training loop, loss functions (BCE, Wasserstein, LSGAN), mode collapse, vanishing gradients, and training tricks. Foundation for understanding all GAN variants.

### **[generation-image.ipynb](notebooks/generation-image.ipynb)**
VAEs and DCGANs for image generation. Latent space exploration.

### **[diffusion-models.ipynb](notebooks/diffusion-models.ipynb)**
Denoising Diffusion (DDPM): forward/reverse diffusion and DDIM sampling.

### **[latent-diffusion.ipynb](notebooks/latent-diffusion.ipynb)**
Latent Diffusion and Stable Diffusion. VAE latents with cross-attention conditioning.

---

## TIER 18: Model Optimization & Interpretability ⏱️ ~6-8 hours

**Making models efficient and understandable.**

### **[model-interpretability.ipynb](notebooks/model-interpretability.ipynb)**
Model interpretability techniques: LIME, SHAP, permutation importance, integrated gradients, and attention visualization. Global vs local explanations for understanding model behavior and individual predictions. Applies to any ML model.

### **[knowledge-distillation.ipynb](notebooks/knowledge-distillation.ipynb)**
Compressing models with teacher-student training and soft targets.

### **[model-pruning.ipynb](notebooks/model-pruning.ipynb)**
Model pruning for neural network compression: unstructured and structured pruning, magnitude-based weight removal, iterative pruning with fine-tuning, and PyTorch's pruning utilities. Achieves 90%+ sparsity with minimal accuracy loss. Complements quantization and distillation for efficient deployment.

### **[1bit-neural-networks.ipynb](notebooks/1bit-neural-networks.ipynb)**
Binary and ternary quantization with straight-through estimator.

---

## TIER 19: Advanced Architectures ⏱️ ~10-14 hours

**Beyond standard feedforward, convolutional, and recurrent networks.**

### **[clip-architecture.ipynb](notebooks/clip-architecture.ipynb)**
CLIP dual encoders with contrastive learning. Zero-shot classification via text.

### **[mixture-of-experts.ipynb](notebooks/mixture-of-experts.ipynb)**
Sparse MoE with gating networks. Efficient scaling to trillion-parameter models.

### 🔴 **[capsule-networks.ipynb](notebooks/capsule-networks.ipynb)**
CapsNets with dynamic routing by agreement. Preserving spatial information.

### 🔴 **[graph-neural-networks.ipynb](notebooks/graph-neural-networks.ipynb)**
GNNs and message passing for graph-structured data.

### 🔴 **[neural-odes.ipynb](notebooks/neural-odes.ipynb)**
Neural ODEs: continuous-depth networks as differential equations.

### 🔴 **[energy-based-neural-networks.ipynb](notebooks/energy-based-neural-networks.ipynb)**
Hopfield Networks and RBMs. Energy functions and contrastive divergence.

---

## TIER 20: Deep Reinforcement Learning ⏱️ ~10-14 hours

**Learning through interaction with environments.**

*Prerequisites: TIER 5A (RL Foundations), TIER 7 (Deep Learning)*

### Policy Methods

### **[rl-policy-gradients.ipynb](notebooks/rl-policy-gradients.ipynb)**
Policy gradients, REINFORCE, Actor-Critic, and PPO basics.

### **[rl-deep-q-networks.ipynb](notebooks/rl-deep-q-networks.ipynb)**
DQN with experience replay and target networks.

### **[rl-advanced-policy-methods.ipynb](notebooks/rl-advanced-policy-methods.ipynb)**
A2C/A3C, SAC, and TD3 for continuous control.

### Search + Learning

### **[wip-alphazero.ipynb](notebooks/wip-alphazero.ipynb)**
AlphaZero: combines Monte Carlo Tree Search with deep neural networks for game mastery through pure self-play. Neural network with policy and value heads guides MCTS exploration, MCTS improves the policy through search, and self-play generates training data. Implements the full algorithm on Tic-Tac-Toe: MCTS with UCB, residual network architecture, iterative self-play training, and evaluation. Shows how search and learning amplify each other's strengths - the virtuous cycle that led to superhuman play in Chess, Go, and Shogi.

### Applications

### **[rlhf-alignment.ipynb](notebooks/rlhf-alignment.ipynb)**
RLHF pipeline: SFT, reward modeling, and PPO optimization. DPO alternative.

### **[rl-model-based.ipynb](notebooks/rl-model-based.ipynb)**
World models, Dyna algorithm, and planning for sample efficiency.

---

## TIER 21: Fascinating Phenomena ⏱️ ~4-6 hours

**Interesting research findings that challenge intuitions.**

### **[adversarial-robustness.ipynb](notebooks/adversarial-robustness.ipynb)**
Adversarial examples, FGSM/PGD attacks, and adversarial training.

### 🔴 **[double-descent.ipynb](notebooks/double-descent.ipynb)**
Double descent: test error decreasing beyond interpolation threshold.

### 🔴 **[grokking.ipynb](notebooks/grokking.ipynb)**
Grokking: sudden generalization after prolonged overfitting on algorithmic tasks.

---

## TIER 22: Alternative Learning Paradigms ⏱️ ~8-12 hours

**Beyond standard supervised learning.**

### Learning Efficiency

### **[meta-learning-few-shot.ipynb](notebooks/meta-learning-few-shot.ipynb)**
MAML for few-shot learning. Learning to learn with inner/outer loops.

### **[active-learning.ipynb](notebooks/active-learning.ipynb)**
Query strategies for selecting data to label: uncertainty, QBC, diversity.

### **[curriculum-learning.ipynb](notebooks/curriculum-learning.ipynb)**
Training from easy to hard. Difficulty metrics and curriculum strategies.

### Multi-Task & Lifelong Learning

### **[multi-task-learning.ipynb](notebooks/multi-task-learning.ipynb)**
MTL with hard/soft parameter sharing. Loss balancing strategies.

### **[continual-learning.ipynb](notebooks/continual-learning.ipynb)**
Lifelong learning without catastrophic forgetting. Replay, EWC, and architecture methods.

### Experimental Methods

### 🔴 **[forward-forward.ipynb](notebooks/forward-forward.ipynb)**
Forward-Forward algorithm: layer-local learning without backpropagation.

### **[neuroevolution-tic-tac-toe.ipynb](notebooks/neuroevolution-tic-tac-toe.ipynb)**
Neuroevolution with genetic algorithms for tic-tac-toe.

---

## BONUS: Fun Projects

**Unique and interesting applications.**

### **[chip8-emulator.ipynb](notebooks/chip8-emulator.ipynb)**
CHIP-8 emulator: fetch-decode-execute cycle and opcodes.

---

## 🗄️ Deprecated / Work in Progress

**Incomplete notebooks not recommended for learning.**

### **[reconstruction-image.ipynb](notebooks/deprecated/reconstruction-image.ipynb)**
Image reconstruction with Vanilla AE, VAE, and VQ-VAE. Replaced by focused vqvae.ipynb notebook.

### **[z2h-05-optimization-pt2-wip.ipynb](notebooks/deprecated/z2h-05-optimization-pt2-wip.ipynb)**
PyTorch experiments with initialization and batch normalization.

### **[z2h-06-backprop-ninja-wip.ipynb](notebooks/deprecated/z2h-06-backprop-ninja-wip.ipynb)**
Manually implementing backward passes for cross-entropy and batch normalization.

### **[z2h-07-wavenet-lm-wip.ipynb](notebooks/deprecated/z2h-07-wavenet-lm-wip.ipynb)**
WaveNet-inspired hierarchical language model.

### **[wip-bit-parity-rnn.ipynb](notebooks/deprecated/wip-bit-parity-rnn.ipynb)**
RNN for bit-parity classification task.

### **[wip-bit-parity-gru.ipynb](notebooks/deprecated/wip-bit-parity-gru.ipynb)**
GRU with update and reset gates for bit-parity classification.

### **[wip-rl-world-model-01-repr.ipynb](notebooks/deprecated/wip-rl-world-model-01-repr.ipynb)**
Compressing gameplay frames with convolutional autoencoders.

### **[wip-rl-world-model-02-dynamics.ipynb](notebooks/deprecated/wip-rl-world-model-02-dynamics.ipynb)**
Learning dynamics models to predict future latent states.

### **[wip-rl-atari-pong-imitation.ipynb](notebooks/deprecated/wip-rl-atari-pong-imitation.ipynb)**
Imitation learning for Atari Pong.

### **[wip-debate-generator.ipynb](notebooks/deprecated/wip-debate-generator.ipynb)**
AI debate simulator with LangChain and multi-agent LLMs.

---

## 🔧 Advanced Usage


### Running Hyperparameter Sweeps

```bash
uv run python run_sweep.py sweeps/config.yaml notebooks/notebook.ipynb --count 10
# See sweeps/*.yaml for config examples (bayes, grid, random)
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

**Note:** This learning progression builds conceptual dependencies systematically. Each tier assumes mastery of previous tiers. Notebooks marked with ✅ have been completed. Notebooks marked with 🔴 are advanced and may require additional background. Total estimated time for complete curriculum: ~180-220 hours.

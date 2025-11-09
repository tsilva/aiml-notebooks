# 🧠 aiml-notebooks

<p align="center">
  <img src="logo.jpg" alt="Logo" width="400"/>
</p>

AI/ML Jupyter notebooks for learning and experimentation. This repository contains educational notebooks covering fundamental to advanced AI/ML concepts, organized in optimal learning order to take you from absolute beginner to advanced practitioner.

## 📖 Overview

This repository provides a comprehensive, hands-on learning path through machine learning and deep learning. Each notebook is designed to build deep intuitions through progressive implementation, starting from absolute foundations and building to state-of-the-art techniques.

**The learning path is organized by conceptual prerequisites** - follow the tier progression for the most effective learning experience. Notebooks marked with ⭐ are foundational and critical for understanding subsequent material.

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. Install uv (if not already installed):
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/tsilva/aiml-notebooks.git
   cd aiml-notebooks
   ```

3. Install dependencies:
   ```bash
   uv sync
   ```

### Running Jupyter

Run JupyterLab directly with uv:
```bash
uv run jupyter lab
```

Or activate the virtual environment first:
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
jupyter lab
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

###  **[basics-numpy-fundamentals.ipynb](notebooks/basics-numpy-fundamentals.ipynb)**
Foundation of numerical computing in Python. Covers array creation, indexing, slicing, broadcasting rules (critical for tensors!), universal functions, and random number generation. Essential prerequisite for understanding PyTorch tensors and all subsequent ML concepts.

###  **[basics-linear-algebra.ipynb](notebooks/basics-linear-algebra.ipynb)** ⭐ **CRITICAL**
Comprehensive introduction to the mathematical language of machine learning. Covers vectors (the building blocks of data), dot products (measuring similarity), norms (different ways to measure size), matrices (transformations and batch operations), linear transformations (what neural network layers do), eigenvalues/eigenvectors (special directions, foundation of PCA), and orthogonality (independence and coordinate systems). Every concept is built from scratch with NumPy, visualized geometrically, and connected to ML applications (embeddings, batch processing, attention mechanisms). Implements power iteration for eigenvalue computation and Gram-Schmidt orthogonalization. Essential foundation for understanding how neural networks, transformers, and virtually all ML algorithms actually work mathematically.

###  **[basics-tensors-operations.ipynb](notebooks/basics-tensors-operations.ipynb)**
Introduction to tensor operations and manipulations in PyTorch, covering shapes, broadcasting, indexing, and fundamental tensor algebra that forms the foundation of deep learning computations.

###  **[basics-tensor-multiplication.ipynb](notebooks/basics-tensor-multiplication.ipynb)**
Deep dive into matrix multiplication, broadcasting rules, and efficient tensor operations. Builds intuition for how neural network layers actually perform computations through matrix operations.

###  **[basics-calculus-refresher.ipynb](notebooks/basics-calculus-refresher.ipynb)** ⭐ **CRITICAL**
Essential calculus foundations for understanding optimization and training. Covers derivatives (rate of change), the chain rule (foundation of backpropagation), partial derivatives (multiple inputs), and gradients (direction of steepest ascent/descent). Builds deep intuition through interactive visualizations of tangent lines, gradient vectors, and gradient descent convergence. Critical prerequisite for understanding how neural networks learn through optimization.

###  **[basics-linear-regression.ipynb](notebooks/basics-linear-regression.ipynb)**
Builds linear regression from scratch to develop deep intuitions about gradient descent, loss functions, and optimization. Compares iterative gradient descent with the analytical closed-form solution, teaching fundamental machine learning concepts that underpin all neural network training.

###  **[basics-metrics.ipynb](notebooks/basics-metrics.ipynb)**
Comprehensive guide to evaluation metrics (accuracy, precision, recall, F1, ROC-AUC, etc.) for classification and regression tasks, teaching when to use each metric and how to interpret them in different contexts.

###  **[basics-probability-distributions.ipynb](notebooks/basics-probability-distributions.ipynb)** ⭐ **CRITICAL**
Foundational introduction to probability distributions that appear throughout machine learning. Covers discrete distributions (PMF: Bernoulli, Categorical, Binomial) and continuous distributions (PDF: Uniform, Gaussian/Normal), teaching mean/variance/sampling with visual intuitions. Explains the Central Limit Theorem and connects distributions to ML concepts like cross-entropy loss, VAEs, and model outputs. Essential prerequisite for understanding loss functions, KL divergence, and generative models.

###  **[basics-loss-functions.ipynb](notebooks/basics-loss-functions.ipynb)**
Provides comprehensive coverage of loss functions (MSE, MAE, cross-entropy, hinge loss) used in machine learning, explaining their mathematical properties and when to use each. Visualizes how different losses handle outliers, class imbalance, and various prediction tasks through interactive experiments. Builds on probability distributions to explain why cross-entropy is the natural loss for classification.

###  **[basics-activation-functions.ipynb](notebooks/basics-activation-functions.ipynb)**
Teaches the fundamental activation functions used in neural networks (ReLU, sigmoid, tanh, etc.) and their properties. Explores how different activation functions affect gradient flow, training dynamics, and model expressiveness through visualizations and hands-on comparisons.

###  **[logistic-regression.ipynb](notebooks/logistic-regression.ipynb)**
Teaches how to build a binary linear classifier using logistic regression on 2D clustered data, covering gradient descent, loss functions (binary cross-entropy), and decision boundary visualization.

###  **[basics-softmax-multiclass.ipynb](notebooks/basics-softmax-multiclass.ipynb)**
Bridges from binary to multiclass classification by teaching the softmax function, one-hot encoding, and categorical cross-entropy loss. Builds a complete multiclass classifier from scratch on the Iris dataset, demonstrating how softmax generalizes sigmoid and why it's the standard approach for multiclass problems. Essential foundation before neural networks.

---

## TIER 2: Core ML Principles

**Understanding how to build models that generalize.**

###  **[basics-overfitting-underfitting.ipynb](notebooks/basics-overfitting-underfitting.ipynb)**
Explores the fundamental machine learning challenge of balancing model capacity using polynomial regression as a teaching tool. Demonstrates how to detect and address underfitting (high bias) and overfitting (high variance) through learning curves, validation strategies, and the bias-variance tradeoff.

###  **[basics-bias-variance-tradeoff.ipynb](notebooks/basics-bias-variance-tradeoff.ipynb)**
Theoretical exploration of the bias-variance decomposition, showing how model error can be broken down into irreducible error, bias, and variance components to guide model selection and complexity decisions.

###  **[basics-train-test-val-split.ipynb](notebooks/basics-train-test-val-split.ipynb)**
Foundational explanation of why we split data into train/validation/test sets, teaching the philosophy behind each split and when to use them. Covers data leakage prevention (preprocessing order, target leakage), temporal split considerations for time-series data, and common mistakes that lead to overly optimistic performance estimates. Essential prerequisite for understanding cross-validation.

###  **[basics-cross-validation.ipynb](notebooks/basics-cross-validation.ipynb)**
Explores three essential cross-validation techniques: K-Fold, Stratified K-Fold, and Time Series Split. Teaches when to use each method, how they provide more reliable performance estimates than single train/test splits, and how to avoid data leakage in temporal data.

###  **[basics-regularization.ipynb](notebooks/basics-regularization.ipynb)**
Explores regularization techniques (L1, L2, dropout, early stopping) that prevent overfitting by constraining model complexity. Shows how regularization adds inductive bias to help models generalize better to unseen data.

###  **[basics-confidence-intervals.ipynb](notebooks/basics-confidence-intervals.ipynb)**
Teaches how to quantify uncertainty in model predictions and evaluation metrics using statistical confidence intervals. Demonstrates bootstrap methods and hypothesis testing to understand whether performance differences between models are statistically significant.

### [x] **[basics-hyperparameter-tuning.ipynb](notebooks/basics-hyperparameter-tuning.ipynb)**
Comprehensive guide to finding the best model configuration through systematic hyperparameter search. Covers grid search (exhaustive exploration), random search (efficient sampling), and Bayesian optimization (smart sequential search). Demonstrates which hyperparameters matter most (learning rate is king!), practical tuning strategies, and when to use each search method. Essential for getting the best performance out of any machine learning model.

###  **[basics-ensemble-methods.ipynb](notebooks/basics-ensemble-methods.ipynb)**
Teaches how combining multiple models produces better predictions than any single model through the "wisdom of crowds" principle. Covers variance reduction through averaging, bagging (bootstrap aggregating), Random Forests (most popular ensemble), boosting methods (AdaBoost, Gradient Boosting) that learn from mistakes sequentially, and stacking (meta-learning to combine models). Demonstrates why model diversity is critical and provides practical guidance on when to use each ensemble approach. Essential practical ML technique used in production systems and competitions.

---

## TIER 3: Data & Optimization

**Making the most of your data and training.**

###  **[basics-data-normalization.ipynb](notebooks/basics-data-normalization.ipynb)** ⭐
Deep dive into data normalization/standardization - min-max scaling vs standardization (z-score), when to use each, why neural networks need normalized inputs, and critically: fitting on train data then applying to test to avoid data leakage. Essential preprocessing skill that directly impacts model convergence and performance.

###  **[basics-dimensionality-reduction.ipynb](notebooks/basics-dimensionality-reduction.ipynb)**
Comprehensive guide to reducing high-dimensional data to lower dimensions for visualization and analysis. Covers the curse of dimensionality (why distances become meaningless in high dimensions), PCA (linear method maximizing variance), t-SNE (non-linear method preserving local structure for beautiful visualizations), and UMAP (modern alternative that's faster and preserves both local and global structure). Demonstrates when to use each method, implements PCA from scratch, explores hyperparameter effects (perplexity for t-SNE, n_neighbors for UMAP), and shows the complete practical workflow: standardize → PCA for understanding → UMAP/t-SNE for visualization. Essential for exploratory data analysis, understanding dataset structure, and preprocessing before machine learning.

###  **[basics-feature-engineering.ipynb](notebooks/basics-feature-engineering.ipynb)**
Covers techniques for creating informative features from raw data including encoding categorical variables, handling missing values, scaling, and domain-specific feature extraction to improve model performance.

###  **[basics-data-augmentation.ipynb](notebooks/basics-data-augmentation.ipynb)**
Teaches data augmentation strategies for artificially expanding training datasets through transformations (rotations, flips, crops for images; synonym replacement for text) to improve model generalization and robustness.

###  **[basics-imbalanced-data.ipynb](notebooks/basics-imbalanced-data.ipynb)**
Addresses the critical problem of class imbalance where one class vastly outnumbers others (e.g., fraud detection: 1% fraud, 99% legitimate). Demonstrates why standard training fails (models predict only majority class), why accuracy is misleading, and teaches proper evaluation metrics (precision, recall, F1-score, confusion matrices). Covers three solution approaches: class weights (penalize minority errors more), resampling (oversampling/undersampling), and SMOTE (synthetic minority sample generation). Essential for real-world applications where balanced datasets are rare.

###  **[basics-mini-batch-gradient-descent.ipynb](notebooks/basics-mini-batch-gradient-descent.ipynb)** ⭐ **CRITICAL**
Teaches the three flavors of gradient descent (full-batch, mini-batch, stochastic) and essential terminology (epochs, steps, iterations, batches). Explains why mini-batching works through gradient noise analysis, memory/computational trade-offs, and practical batch size selection. Critical prerequisite for understanding optimizers, as all modern optimizers assume mini-batch training with noisy gradients.

###  **[basics-optimizers.ipynb](notebooks/basics-optimizers.ipynb)**
Deep dive into optimization algorithms (SGD, Momentum, RMSprop, Adam) that train neural networks, building intuition through visualization of their paths through loss landscapes. Explains how each optimizer addresses specific challenges like ravines, saddle points, and different parameter scales, with practical guidance on when to use each.

###  **[basics-learning-rate-schedules.ipynb](notebooks/basics-learning-rate-schedules.ipynb)**
Teaches how to improve training by dynamically adjusting learning rates over time using schedules like step decay, exponential decay, and cosine annealing. Demonstrates why starting with high learning rates and gradually reducing them leads to better convergence, with warmup techniques for large models.

---

## TIER 4: Deep Learning Foundations

**Now we can start with neural networks!**

###  **[basics-neural-network-fundamentals.ipynb](notebooks/basics-neural-network-fundamentals.ipynb)**
Foundation for understanding what neural networks actually are. Covers the core building blocks: single neurons (perceptrons), layers as collections of neurons, and networks as stacked layers. Explains forward pass computation, architecture terminology (width, depth, parameters), why we stack layers for hierarchical feature learning, and provides intuition for the Universal Approximation Theorem. Essential prerequisite for understanding backpropagation and training.

###  **[basics-computational-graphs.ipynb](notebooks/basics-computational-graphs.ipynb)** ⭐ **CRITICAL**
Teaches how to represent any computation as a directed acyclic graph (DAG) where nodes are values/operations and edges represent data flow. Builds visual intuition for how graphs naturally encode the chain rule, making backpropagation obvious. Shows how PyTorch's automatic differentiation builds and traverses these graphs during forward and backward passes. **Essential foundation before backprop!**

###  **[z2h-01-backprop.ipynb](notebooks/z2h-01-backprop.ipynb)** ⭐ **CRITICAL**
Teaches backpropagation from scratch by building a micrograd-style autograd engine that tracks computational graphs, calculates gradients using the chain rule, and trains neural networks using gradient descent. **Must understand this deeply before proceeding!**

###  **[basics-pytorch-fundamentals.ipynb](notebooks/basics-pytorch-fundamentals.ipynb)**
Bridges conceptual neural network understanding with PyTorch's practical implementation tools. Covers autograd (automatic differentiation and `.backward()`), nn.Parameter (how PyTorch tracks trainable parameters), nn.Module (foundation for building components), the five-step training loop structure (forward, loss, zero gradients, backward, update), and building custom layers. Essential for understanding how PyTorch automates the backpropagation you just learned to implement manually.

###  **[basics-gradient-stability.ipynb](notebooks/basics-gradient-stability.ipynb)**
Explores the vanishing and exploding gradient problems that plague deep neural networks, demonstrating how gradients can shrink or grow exponentially during backpropagation through many layers and why this matters for training.

###  **[basics-gradient-clipping.ipynb](notebooks/basics-gradient-clipping.ipynb)**
Teaches gradient clipping as a solution to exploding gradients, showing how to cap gradient magnitudes to stabilize training in recurrent networks and deep architectures.

###  **[basics-gradient-normalization.ipynb](notebooks/basics-gradient-normalization.ipynb)**
Covers gradient normalization techniques that rescale gradients to have consistent magnitude across training, improving optimization stability especially in networks with varying layer depths.

###  **[basics-batch-normalization.ipynb](notebooks/basics-batch-normalization.ipynb)**
Explains batch normalization, a technique that normalizes layer inputs to stabilize and accelerate training. Covers the internal covariate shift problem, demonstrates how BatchNorm maintains stable distributions across layers, and shows its impact on training speed and performance.

###  **[basics-layer-normalization.ipynb](notebooks/basics-layer-normalization.ipynb)**
Introduces layer normalization as an alternative to batch normalization that normalizes across features rather than batch dimensions, making it more suitable for recurrent networks and transformers.

---

## TIER 5: First Neural Networks

**Simple neural networks for language modeling.**

###  **[z2h-02-bigram-lm.ipynb](notebooks/z2h-02-bigram-lm.ipynb)**
Introduces character-level bigram language models by manually building probability distributions from character pair frequencies, then recreating the same model using PyTorch with backpropagation and gradient descent.

###  **[z2h-03-mlp-lm.ipynb](notebooks/z2h-03-mlp-lm.ipynb)**
Builds a multi-layer perceptron (MLP) language model for character-level text generation, showing how adding hidden layers and nonlinearities enables the model to capture more complex patterns than simple bigrams.

###  **[z2h-04-optimization-pt1.ipynb](notebooks/z2h-04-optimization-pt1.ipynb)**
Focuses on neural network optimization techniques including weight initialization strategies (Xavier, Kaiming), analyzing activation and gradient flow through layers, and understanding how these choices critically affect training performance.

###  **[z2h-05-optimization-pt2-wip.ipynb](notebooks/z2h-05-optimization-pt2-wip.ipynb)** *(Work in Progress)*
Continues optimization topics by migrating code to PyTorch and conducting experiments with initialization strategies and batch normalization to understand their impact on data and gradient flow through neural networks.

###  **[z2h-06-backprop-ninja-wip.ipynb](notebooks/z2h-06-backprop-ninja-wip.ipynb)** *(Work in Progress)*
Advanced backpropagation practice where students manually implement backward passes for complex operations (cross-entropy, batch normalization) to deeply understand gradient computation in neural networks.

###  **[basics-debugging-neural-networks.ipynb](notebooks/basics-debugging-neural-networks.ipynb)**
Practical guide to debugging neural networks systematically. Covers essential sanity checks (overfitting small batches), common failure modes (dying ReLU, NaN losses, mode collapse), gradient flow visualization, and learning curve interpretation (overfitting, underfitting, instability). Teaches a complete debugging workflow from identifying problems to testing fixes methodically.

---

## TIER 6: Convolutional Neural Networks

**Computer vision fundamentals.**

###  **[basics-tensors-convolution.ipynb](notebooks/basics-tensors-convolution.ipynb)**
Teaches convolutional operations from first principles, showing how filters slide across inputs to extract features. Covers padding, stride, dilation, multi-channel convolutions, and transposed convolutions. Essential foundation for understanding CNNs and image processing architectures.

###  **[basics-pooling-operations.ipynb](notebooks/basics-pooling-operations.ipynb)**
Comprehensive guide to pooling operations (MaxPooling, AveragePooling, Global pooling) that downsample feature maps produced by convolutional layers. Explains why downsampling is critical (computational efficiency, translation invariance, receptive field expansion), demonstrates spatial invariance through interactive examples, and teaches output size calculations. Covers common pitfalls and when to use each pooling type. Essential for building complete CNN architectures.

###  **[classification-image.ipynb](notebooks/classification-image.ipynb)**
Demonstrates building image classifiers with configurable architectures (CNN or MLP) on multiple datasets (CIFAR-10, MNIST, Fashion-MNIST), using PyTorch Lightning for training with automatic checkpointing and W&B logging.

###  **[basics-residual-connections.ipynb](notebooks/basics-residual-connections.ipynb)** ⭐ **REVOLUTIONARY**
Revolutionary architectural innovation that enabled training networks with 100+ layers. Teaches the vanishing gradient problem in deep networks, skip connections (residual blocks), and why they work through gradient highways and ensemble interpretation. Implements ResNet blocks from scratch and demonstrates the famous result: plain networks degrade with depth, but ResNets improve. Essential foundation for understanding ResNet, Transformers, U-Net, and all modern deep architectures.
###  **[basics-transfer-learning.ipynb](notebooks/basics-transfer-learning.ipynb)** ⭐ **CRITICAL - Used in 90% of Real Applications**
Teaches transfer learning, the most practical deep learning technique for real-world applications. Covers pretrained models (ResNet on ImageNet), feature extraction (freezing layers) vs fine-tuning (unfreezing with low learning rates), when to use each approach based on dataset size and domain similarity, differential learning rates for optimal fine-tuning, and hands-on comparison on CIFAR-10. Essential for working with limited data and achieving state-of-the-art results quickly.

###  **[basics-knowledge-distillation.ipynb](notebooks/basics-knowledge-distillation.ipynb)** ⭐ **CRITICAL - Used in Production**
One of the most practical techniques for deploying models in production. Teaches how to compress large, accurate models into small, fast models with minimal accuracy loss. Covers teacher-student training framework, soft targets vs hard labels, temperature scaling for controlling prediction softness, response-based distillation (matching outputs), feature-based distillation (matching intermediate layers), and practical deployment considerations. Demonstrates achieving 10-100x speedup with <3% accuracy drop. Essential for mobile deployment, real-time inference, and cost-effective serving at scale.

###  **[basics-unet-architecture.ipynb](notebooks/basics-unet-architecture.ipynb)** ⭐ **CRITICAL for Segmentation**
Teaches U-Net, the gold-standard architecture for semantic segmentation tasks (pixel-wise classification). Covers the complete encoder-decoder structure with skip connections, transpose convolutions for learnable upsampling, and why U-Net's concatenation-based skip connections (different from ResNet) preserve fine-grained spatial information. Implements U-Net from scratch, demonstrates training on synthetic circle segmentation, and includes ablation study proving skip connections are critical for precise localization. Essential for medical imaging (tumor/organ segmentation), autonomous driving (lane detection), satellite imagery, and any task requiring pixel-level predictions. Shows why U-Net works with limited data and revolutionized biomedical image analysis.
###  **[grad-cam-visualization.ipynb](notebooks/grad-cam-visualization.ipynb)** ⭐ **CRITICAL for Model Interpretability**
Essential technique for understanding and debugging CNN decisions through visualization. Teaches Grad-CAM (Gradient-weighted Class Activation Mapping) to generate class-discriminative heatmaps showing which image regions influence predictions. Covers the complete algorithm (forward pass, gradient computation, weighted combination), implements from scratch with PyTorch hooks, and compares with other visualization methods (vanilla gradients, guided backpropagation, guided Grad-CAM). Demonstrates practical applications: debugging misclassifications, detecting dataset biases, building trust in model decisions. Critical for production deployment where model interpretability is required (medical diagnosis, autonomous vehicles, etc.).

###  **[object-detection-yolo.ipynb](notebooks/object-detection-yolo.ipynb)** ⭐ **Core Computer Vision Application**
Comprehensive introduction to object detection using YOLO (You Only Look Once), the foundational single-stage detector. Teaches detection fundamentals (localization + classification, bounding boxes, IoU), YOLO's grid-based architecture with anchor boxes, the multi-task loss function (localization + objectness + classification), and Non-Maximum Suppression (NMS) for duplicate removal. Implements simplified YOLO from scratch, demonstrates on geometric shapes dataset, and compares with two-stage detectors (R-CNN family). Essential for understanding modern object detection systems used in autonomous vehicles, surveillance, robotics, and real-time video analysis.

---

## TIER 7: Information Theory & Embeddings

**Important concepts for advanced topics.**

###  **[basics-tokenization-methods.ipynb](notebooks/basics-tokenization-methods.ipynb)** ⭐ **CRITICAL for NLP**
Foundational introduction to tokenization - the first step in all NLP pipelines. Compares four major approaches: character-level (tiny vocab, long sequences), word-level (semantic but OOV problems), BPE (GPT's method, frequency-based subwords), and WordPiece (BERT's method, likelihood-based subwords). Builds each tokenizer from scratch to understand merge algorithms, demonstrates OOV handling, morphological sharing, and the fundamental vocab-size vs sequence-length tradeoff. Shows why subword tokenization dominates modern NLP by balancing vocabulary size, OOV robustness, and computational efficiency. Essential prerequisite for understanding how models like GPT and BERT process text before embeddings.

###  **[basics-kl-divergence.ipynb](notebooks/basics-kl-divergence.ipynb)** ⭐ **CRITICAL for VAEs & RL**
Teaches Kullback-Leibler (KL) divergence as a fundamental information-theoretic measure of how one probability distribution differs from another. Builds intuition from first principles through information content, entropy, and cross-entropy, explaining the asymmetry of KL divergence, forward vs reverse KL behavior (mode-seeking vs mode-covering), and critical applications in VAEs, reinforcement learning policy optimization (TRPO/PPO), variational inference, and distribution matching.

###  **[basics-embeddings.ipynb](notebooks/basics-embeddings.ipynb)**
Introduces the concept of learned dense vector representations for discrete objects (words, items, users), showing how embeddings capture semantic relationships and enable neural networks to process categorical data effectively.

###  **[word2vec-from-scratch.ipynb](notebooks/word2vec-from-scratch.ipynb)**
Implements Word2Vec using the skip-gram model with negative sampling to learn word embeddings from scratch. Demonstrates the distributional hypothesis ("you shall know a word by the company it keeps") and shows how vector arithmetic captures semantic relationships (king - man + woman ≈ queen). Essential practical implementation of embedding concepts that revolutionized NLP before transformers.

###  **[siamese-networks.ipynb](notebooks/siamese-networks.ipynb)** ⭐ **CRITICAL for Similarity Learning**
Teaches Siamese Networks, a foundational architecture for learning similarity between inputs using shared weights and contrastive loss. Covers the complete similarity learning pipeline: distance metrics (Euclidean, cosine), contrastive loss that pulls similar pairs together and pushes dissimilar pairs apart, building twin networks with shared weights, and training on MNIST digit pairs. Demonstrates one-shot learning (recognizing new classes from single examples), visualizes learned embedding spaces with t-SNE showing semantic clustering, and introduces triplet loss as an advanced alternative. Essential foundation for face verification, signature verification, metric learning, and few-shot learning applications used extensively in production systems.

---

## TIER 8: Unsupervised Learning Basics

**Learning without labels.**

###  **[kmeans-from-scratch.ipynb](notebooks/kmeans-from-scratch.ipynb)**
Builds the k-means clustering algorithm from first principles, showing how it partitions data through iterative assignment and update steps. Covers distance metrics, initialization strategies (k-means++), convergence guarantees, and the elbow method for choosing k.

###  **[basics-gaussian-mixture-models.ipynb](notebooks/basics-gaussian-mixture-models.ipynb)**
Extends k-means to probabilistic soft clustering using Gaussian Mixture Models (GMMs). Teaches the Expectation-Maximization (EM) algorithm, soft vs hard cluster assignments with uncertainty quantification, different covariance types (spherical, diagonal, full) for capturing cluster shapes, the mathematical connection between GMMs and k-means, and model selection using BIC/AIC instead of the elbow method. Essential for understanding probabilistic models and a foundation for VAEs.

###  **[basics-autoencoders.ipynb](notebooks/basics-autoencoders.ipynb)** ⭐ **CRITICAL - Foundation for VAEs**
Introduces autoencoders as a fundamental unsupervised learning architecture that learns to compress data into a compact latent representation and reconstruct it. Teaches the encoder-decoder architecture, bottleneck principle, reconstruction loss (MSE vs BCE), and training from scratch on MNIST. Visualizes learned latent spaces with t-SNE/PCA showing semantic clustering, performs smooth interpolation between images in latent space, and demonstrates practical applications including denoising and anomaly detection. Essential prerequisite for understanding Variational Autoencoders (VAEs), generative models, and representation learning. Builds intuition for why the bottleneck forces networks to learn meaningful features.

---

## TIER 9: Recurrent Neural Networks

**Sequential data and memory.**

###  **[rnn-from-scratch.ipynb](notebooks/rnn-from-scratch.ipynb)**
A complete implementation of Recurrent Neural Networks (RNNs) for character-level language modeling. Teaches how networks maintain "memory" through hidden states, the mechanics of backpropagation through time (BPTT), and why RNNs suffer from vanishing gradients.

###  **[lstm-from-scratch.ipynb](notebooks/lstm-from-scratch.ipynb)**
Builds LSTM networks from scratch using only NumPy to develop deep understanding of gate mechanisms (forget, input, output), cell state vs hidden state, and how LSTMs solve the vanishing gradient problem for long-range dependencies.

###  **[wip-bit-parity-rnn.ipynb](notebooks/wip-bit-parity-rnn.ipynb)** *(Work in Progress)*
Teaches the fundamentals of recurrent neural networks (RNNs) through the bit-parity classification task, demonstrating building an RNN from scratch to classify binary sequences as having odd or even parity.

###  **[wip-bit-parity-gru.ipynb](notebooks/wip-bit-parity-gru.ipynb)** *(Work in Progress)*
Teaches the fundamentals of Gated Recurrent Units (GRUs) through the bit-parity classification task, implementing a GRU cell from scratch with explicit update gates, reset gates, and candidate hidden states.

---

## TIER 10: Attention & Transformers

**Modern sequence modeling - the revolution.**

###  **[basics-attention-mechanism.ipynb](notebooks/basics-attention-mechanism.ipynb)** ⭐ **FOUNDATIONAL CONCEPT**
Introduces the attention mechanism that revolutionized deep learning, explaining how models can "focus" on relevant parts of input sequences. Builds intuition through step-by-step implementation of scaled dot-product attention (queries, keys, values), attention weights, and the attention function. Essential foundation before understanding how attention is applied in different architectures.

###  **[seq2seq-with-attention.ipynb](notebooks/seq2seq-with-attention.ipynb)**
Introduces sequence-to-sequence (seq2seq) encoder-decoder architecture for transforming one sequence into another (e.g., string reversal, translation). Demonstrates the **bottleneck problem** in basic RNN encoder-decoders where information gets lost in long sequences, then shows how the **attention mechanism** solves this. Shows how attention allows the decoder to dynamically access all encoder hidden states through learned alignment weights. Includes extensive attention visualization showing which input positions influence each output token, making the model's decision-making process interpretable. Essential bridge from RNNs to modern attention-based architectures, showing why attention revolutionized NLP.

###  **[basics-positional-encodings.ipynb](notebooks/basics-positional-encodings.ipynb)** ⭐ **CRITICAL for Transformers**
Comprehensive exploration of how transformers understand sequence order. Demonstrates that self-attention is permutation invariant (cannot see order), then teaches all major positional encoding approaches: **Sinusoidal** (original Transformer's wavelength-based encoding with relative position properties), **Learned Embeddings** (GPT-style trainable positions), **Relative Position Bias** (T5's distance-based approach), **RoPE** (Rotary Position Embeddings used in LLaMA for length extrapolation through rotation in complex plane), and **ALiBi** (simplest approach with linear attention bias). Implements each from scratch with extensive visualizations (heatmaps, frequency patterns, rotation angles). Compares all methods through training experiments on character-level language modeling, showing when to use each approach. Essential foundation for understanding why modern LLMs favor RoPE and ALiBi for handling long contexts.

###  **[transformer-from-scratch.ipynb](notebooks/transformer-from-scratch.ipynb)** ⭐ **CRITICAL for Modern AI**
Builds a Transformer architecture incrementally from basic building blocks to a complete model. Covers the self-attention mechanism (the core innovation), multi-head attention, positional encodings, and how Transformers process sequences in parallel unlike RNNs. Implements the full encoder-decoder architecture that forms the foundation for BERT (encoder-only) and GPT (decoder-only) variants.

###  **[gpt-architecture.ipynb](notebooks/gpt-architecture.ipynb)** ⭐ **Essential Generative Architecture**
Complete introduction to GPT (Generative Pre-trained Transformer), the foundational architecture powering ChatGPT and modern language models. Teaches autoregressive generation, causal (unidirectional) masking to prevent future information leakage, BPE tokenization, and the decoder-only transformer architecture. Implements from scratch on character-level language modeling, demonstrates next-token prediction training objective, and covers all sampling strategies: greedy (deterministic), temperature (controlling randomness), top-k (limiting to k most likely), and nucleus/top-p (dynamic probability mass). Compares with BERT's bidirectional approach and explains why causal masking is essential for generation. Foundation for understanding GPT-2, GPT-3, GPT-4, and all autoregressive language models.

###  **[bert-architecture.ipynb](notebooks/bert-architecture.ipynb)** ⭐ **Essential Modern NLP Architecture**
Comprehensive guide to BERT (Bidirectional Encoder Representations from Transformers), the breakthrough model that revolutionized NLP through bidirectional context understanding. Teaches masked language modeling (MLM) with 80/10/10 masking strategy, next sentence prediction (NSP), WordPiece tokenization with special tokens ([CLS], [SEP], [MASK], [PAD]), and the complete BERT architecture (embeddings, multi-head self-attention, feed-forward layers). Implements from scratch, demonstrates on sentiment analysis using [CLS] token, visualizes bidirectional vs unidirectional attention patterns, and compares with GPT-style models. Essential for understanding modern pre-trained language models like RoBERTa, ALBERT, and encoder-based architectures.

###  **[vision-transformers.ipynb](notebooks/vision-transformers.ipynb)**
Demonstrates how transformers conquered computer vision through Vision Transformers (ViT). Teaches patch embeddings (splitting images into sequences), 2D positional encodings for spatial information, and building a complete ViT from scratch. Compares ViT with CNNs on CIFAR-10, visualizes learned attention patterns to see what the model focuses on, and explores the trade-offs between inductive bias (CNNs) and learned spatial understanding (ViT). Shows why transformers are now a universal architecture across text, vision, and beyond.

###  **[lora-peft.ipynb](notebooks/lora-peft.ipynb)** ⭐ **CRITICAL for Practical LLM Usage**
Comprehensive guide to Parameter-Efficient Fine-Tuning (PEFT), the breakthrough technique that makes LLM fine-tuning accessible on consumer hardware. Teaches LoRA (Low-Rank Adaptation) through the core insight that weight updates during fine-tuning are low-rank, enabling 100-1000x parameter reduction (train only 0.1-1% of weights). Covers the LoRA equation W = W₀ + BA with frozen pretrained weights and trainable low-rank matrices, initialization strategies (zero-init B ensures starting from pretrained), rank/alpha hyperparameter selection, and merging for zero-overhead inference. Implements LoRALinear layer from scratch, builds complete LoRA transformer, and demonstrates sentiment classification training only 2% of parameters. Explains QLoRA (4-bit quantization + LoRA) enabling 70B model fine-tuning on single GPUs through INT4 base model with FP16 adapters. Compares alternative PEFT methods: Prefix Tuning (learnable prefix tokens), Prompt Tuning (soft prompts, fewest parameters), and Adapters (bottleneck layers). Shows practical HuggingFace PEFT library usage and multi-task adapter switching. Essential for anyone fine-tuning modern LLMs (GPT, LLaMA, BERT) with limited compute, enabling personalized models and domain adaptation.

###  **[z2h-07-wavenet-lm-wip.ipynb](notebooks/z2h-07-wavenet-lm-wip.ipynb)** *(Work in Progress)*
Implements a WaveNet-inspired hierarchical language model using custom layers (embeddings, batch normalization, sequential flattening) to process longer character contexts more efficiently than flat architectures.

---

## TIER 11: Text Applications

**Practical NLP tasks end-to-end.**

###  **[classification-text.ipynb](notebooks/classification-text.ipynb)**
Explores sentiment analysis on IMDb movie reviews using three progressive approaches: TF-IDF + Logistic Regression (baseline), LSTM neural networks, and fine-tuned BERT, comparing their performance and trade-offs.

###  **[generation-text.ipynb](notebooks/generation-text.ipynb)**
Comprehensively covers text generation from bigram statistical models through neural approaches (MLP, RNN, LSTM) to transformers, including sampling strategies (temperature, top-k, nucleus) and latent space manipulation.

###  **[reconstruction-text.ipynb](notebooks/reconstruction-text.ipynb)**
Teaches sequence-to-sequence autoencoders for text reconstruction using encoder-decoder architectures, demonstrating latent space interpolation, teacher forcing training strategies, and applications like semantic similarity measurement.

###  **[denoising-text.ipynb](notebooks/denoising-text.ipynb)**
Demonstrates text correction and denoising techniques using seq2seq models with attention and transformers, including noise injection strategies, edit distance baselines, and evaluation metrics (CER/WER).

###  **[ner-token-classification.ipynb](notebooks/ner-token-classification.ipynb)** ⭐ **Core NLP Task**
Comprehensive guide to Named Entity Recognition (NER), a fundamental token-level classification task that identifies and classifies entities (person names, organizations, locations) in text. Teaches the BIO tagging scheme (Beginning-Inside-Outside) for representing multi-word entities, explains why it's superior to simple labeling for consecutive entities. Implements four progressive approaches from scratch: rule-based baseline (gazetteers and heuristics), BiLSTM (bidirectional context), BiLSTM-CRF (sequence constraints via Conditional Random Fields for valid tag transitions), and BERT (pre-trained contextualized embeddings). Deep dive into CRF theory: transition matrices, forward algorithm for partition functions, and Viterbi decoding for optimal sequences. Handles BERT's subword tokenization challenge through careful label alignment (only predict on first subword, ignore continuations). Teaches proper entity-level F1 evaluation (span-based metrics) vs misleading token accuracy. Covers practical considerations: nested entities, few-shot/zero-shot NER with LLMs, and active learning for efficient annotation. Essential foundation for information extraction, question answering, and knowledge graph construction.

###  **[qa-extractive.ipynb](notebooks/qa-extractive.ipynb)** ⭐ **Core NLP Task - SQuAD-style QA**
Comprehensive guide to extractive question answering, the task of finding answer spans within context paragraphs (foundation for search engines and virtual assistants). Covers the SQuAD dataset format, TF-IDF baselines showing limitations of heuristics, BiDAF architecture with bidirectional attention flow (context-to-query and query-to-context attention mechanisms) implemented from scratch, and BERT for production QA (input formatting with [CLS]/[SEP] tokens, span prediction via start/end logits, proper decoding with validity constraints). Teaches critical evaluation metrics (Exact Match for strict matching, F1 for token overlap), handling impossible questions (SQuAD 2.0 using CLS token scores), and advanced topics including multi-hop reasoning, open-domain QA (retrieval + reading), and conversational QA. Demonstrates complete pipeline from data preparation to prediction postprocessing. Essential for understanding modern reading comprehension systems, information extraction, and how transformers excel at span prediction tasks.

###  **[rag-retrieval-augmented.ipynb](notebooks/rag-retrieval-augmented.ipynb)** ⭐ **CRITICAL - Essential for Production LLM Applications**
Comprehensive guide to Retrieval-Augmented Generation (RAG), the foundational technique for building production LLM applications with up-to-date and domain-specific knowledge. Teaches the complete RAG pipeline: document ingestion and chunking strategies (sentence-based with overlap), embedding generation using sentence transformers, vector storage and semantic search (cosine similarity implemented from scratch), prompt augmentation templates, and integration with language models. Covers three retrieval approaches: dense retrieval (embeddings for semantic similarity), sparse retrieval (BM25 for keyword matching, implemented from scratch), and hybrid retrieval (combining both). Demonstrates re-ranking retrieved chunks for relevance, advanced techniques like HyDE (Hypothetical Document Embeddings), and proper evaluation metrics (Recall@k, Precision@k, MRR). Teaches production considerations: vector database integration with FAISS for large-scale systems, chunking size optimization, caching strategies, and cost optimization. Essential for building question answering systems, document search, chatbots with custom knowledge bases, and any application requiring LLMs to reason over private or current data beyond their training cutoff.

###  **[time-series-forecasting.ipynb](notebooks/time-series-forecasting.ipynb)** ⭐ **Critical Real-World Application**
Comprehensive guide to time series forecasting with neural networks, one of the most important real-world ML applications (finance, weather, energy, demand prediction). Teaches time series fundamentals (autoregression, lookback windows, forecast horizons, stationarity, seasonality), proper temporal train/val/test splitting to prevent data leakage, and sliding window sequence creation. Implements three progressive architectures from scratch: MLP baseline (feedforward), LSTM (sequential processing with memory), and Transformer (attention-based). Covers appropriate evaluation metrics (MAE, RMSE, MAPE), demonstrates multi-step forecasting, and discusses critical challenges (distribution shift, long-term dependencies). Essential for any production ML application dealing with temporal data.

---

## TIER 12: Generative Models for Images

**Learning to generate and reconstruct images.**

###  **[reconstruction-image.ipynb](notebooks/reconstruction-image.ipynb)**
Explores image reconstruction through three autoencoder variants (Vanilla, VAE, VQ-VAE) on MNIST, demonstrating compression, latent space analysis, and applications like denoising and anomaly detection.

###  **[generation-image.ipynb](notebooks/generation-image.ipynb)** ⭐ **VAEs & GANs**
Teaches image generation using Variational Autoencoders (VAE) and Deep Convolutional GANs (DCGAN) on MNIST, covering latent space exploration, interpolation, and the trade-offs between explicit vs implicit generative models.

###  **[diffusion-models.ipynb](notebooks/diffusion-models.ipynb)** ⭐ **CRITICAL - State-of-the-Art Generation**
Teaches Denoising Diffusion Probabilistic Models (DDPM), the breakthrough technology powering Stable Diffusion, DALL-E 2, and Midjourney. Covers the forward diffusion process (progressive noise addition), reverse diffusion (learning to denoise), U-Net architecture with time embeddings, DDPM training objective (noise prediction), DDIM sampling (10-20x faster generation), and connections to score-based models. Demonstrates why diffusion models achieve state-of-the-art quality with stable training, combining the best aspects of VAEs (stability) and GANs (quality). Essential for understanding modern generative AI systems that are revolutionizing creative applications.

###  **[latent-diffusion.ipynb](notebooks/latent-diffusion.ipynb)** ⭐ **CRITICAL - How Stable Diffusion Actually Works**
Comprehensive guide to Latent Diffusion Models, the architecture powering Stable Diffusion and modern text-to-image generation. Extends pixel-space DDPMs by operating in compressed latent space for dramatic efficiency gains (8× compression, 64× faster). Teaches the complete pipeline: VAE encoder/decoder for perceptual compression (image ↔ latent conversion), diffusion process operating on latent vectors instead of pixels (why latents are more semantic), U-Net with cross-attention for text conditioning (how text guides image generation), and classifier-free guidance (CFG formula for quality improvement). Implements from scratch on MNIST with text conditioning, demonstrates DDIM sampling for fast generation (50 steps vs 1000), and provides complete Stable Diffusion architecture breakdown showing how CLIP text encoder, latent diffusion U-Net, and VAE decoder interact. Covers extensions: image-to-image (img2img), inpainting, ControlNet (structure control), and LoRA fine-tuning. Essential for understanding why Stable Diffusion revolutionized generative AI through computational efficiency and controllability, enabling high-resolution text-to-image generation on consumer hardware.

###  **[denoising-image.ipynb](notebooks/denoising-image.ipynb)**
Focuses on removing noise from corrupted images using convolutional autoencoders and denoising autoencoders (DAE), teaching how models learn robust representations by reconstructing clean data from noisy inputs.

---

## TIER 13: Advanced Architectures

**Beyond standard feedforward, convolutional, and recurrent networks.**

###  **[clip-architecture.ipynb](notebooks/clip-architecture.ipynb)** ⭐ **CRITICAL - Foundation for Multimodal AI**
Comprehensive guide to CLIP (Contrastive Language-Image Pre-training), the breakthrough model that connects vision and language in a shared embedding space. Teaches the dual encoder architecture (image encoder + text encoder projecting to common space), contrastive learning objective (InfoNCE loss maximizing similarity for matching image-text pairs), and zero-shot classification through natural language descriptions. Implements simplified CLIP from scratch on MNIST with generated captions, demonstrating how to align visual and textual representations through symmetric contrastive training. Covers the N×N similarity matrix approach (efficient batch-level negatives), learnable temperature parameter for controlling task difficulty, and zero-shot transfer where new classes need only text descriptions. Visualizes the learned multimodal embedding space with t-SNE, implements text-to-image search, and explores compositional understanding across different phrasings. Essential foundation for understanding DALL-E, Stable Diffusion, GPT-4 Vision, Flamingo, and all modern vision-language models. Revolutionized computer vision by replacing expensive labeled datasets with abundant image-text pairs from the web, enabling unprecedented generalization and natural language interfaces for visual tasks.

###  **[capsule-networks.ipynb](notebooks/capsule-networks.ipynb)**
Teaches Capsule Networks (CapsNets), Geoffrey Hinton's alternative to CNNs that addresses fundamental limitations in how CNNs handle part-whole spatial relationships. Covers the core problems with pooling (loses spatial information), what capsules are (vector outputs encoding entity properties), dynamic routing by agreement (capsules communicate based on agreement rather than fixed weights), and equivariance vs invariance. Implements complete CapsNet from scratch on MNIST with squash activation, margin loss, and decoder-based reconstruction. Demonstrates dimension perturbation to visualize what each capsule dimension encodes and shows superior robustness to affine transformations. Essential for understanding alternatives to standard CNN architectures and the importance of preserving spatial information.
###  **[mixture-of-experts.ipynb](notebooks/mixture-of-experts.ipynb)** ⭐ **CRITICAL for Modern LLMs**
Teaches Mixture of Experts (MoE), the breakthrough architecture enabling efficient scaling to trillion-parameter models. Covers sparse activation (only k out of N experts process each input), gating networks that learn to route inputs to relevant experts, load balancing to ensure equal expert usage, and integration with Transformers by replacing FFN layers. Demonstrates how MoE achieves 10x more model capacity with similar compute cost through specialization. Explains why modern LLMs like Mixtral 8x7B and GPT-4 use MoE, showing how Mixtral has 47B parameters but only ~13B active per token. Essential for understanding how massive models scale efficiently and why MoE is the future of LLM architecture.

###  **[graph-neural-networks.ipynb](notebooks/graph-neural-networks.ipynb)**
Introduces Graph Neural Networks (GNNs) for learning on graph-structured data like social networks and molecules. Teaches the message passing framework where nodes aggregate information from neighbors, demonstrated through semi-supervised node classification.

###  **[neural-odes.ipynb](notebooks/neural-odes.ipynb)**
Teaches Neural Ordinary Differential Equations (Neural ODEs) - continuous-depth networks that model transformations as flows defined by differential equations. Builds intuition from ResNets as discrete Euler methods, explains the adjoint method for memory-efficient backpropagation, and demonstrates adaptive computation. Elegant mathematical framework connecting neural networks to dynamical systems, with applications in continuous normalizing flows, time series, and physics-informed learning.

###  **[energy-based-neural-networks.ipynb](notebooks/energy-based-neural-networks.ipynb)**
Explores energy-based models (EBMs) including Hopfield Networks and Restricted Boltzmann Machines (RBMs). Teaches how energy functions assign low energy to "good" configurations, the Gibbs distribution, and contrastive divergence for training.

---

## TIER 14: Fascinating Phenomena

**Interesting research findings that challenge intuitions.**

###  **[adversarial-robustness.ipynb](notebooks/adversarial-robustness.ipynb)** ⭐ **CRITICAL for Deployment**
Teaches adversarial examples and robustness - how neural networks can be fooled by imperceptible perturbations. Covers FGSM and PGD attacks (generating adversarial examples), adversarial training (defending through worst-case training), and certified defenses (provable robustness guarantees). Demonstrates the fundamental fragility of neural networks and why this matters critically for deploying ML in security-sensitive applications like autonomous vehicles, face recognition, malware detection, and medical diagnosis. Essential practical consideration for reliable, safe AI systems.

###  **[double-descent.ipynb](notebooks/double-descent.ipynb)**
Explores the double descent phenomenon where test error decreases again after the interpolation threshold, challenging classical bias-variance tradeoff intuitions and demonstrating why overparameterized models can generalize better than expected.

###  **[grokking.ipynb](notebooks/grokking.ipynb)**
Demonstrates the grokking phenomenon where neural networks suddenly transition from memorization to generalization after prolonged training on algorithmic tasks (modular arithmetic), showing how models can achieve perfect generalization thousands of epochs after overfitting.

---

## TIER 15: Alternative Training Paradigms

**Beyond standard supervised learning - self-supervised, meta-learning, and alternative training methods.**

###  **[meta-learning-few-shot.ipynb](notebooks/meta-learning-few-shot.ipynb)** ⭐ **Important Modern Paradigm**
Comprehensive introduction to meta-learning ("learning to learn") and few-shot learning, where models adapt to new tasks with minimal examples. Teaches the N-way K-shot classification problem, support/query set structure, and MAML (Model-Agnostic Meta-Learning) algorithm with inner loop (task adaptation) and outer loop (meta-optimization). Explains second-order gradients (gradient-through-gradient) that enable meta-learning. Implements MAML from scratch on sine wave regression, demonstrates rapid adaptation, and compares with Prototypical Networks (metric-based meta-learning). Contrasts with transfer learning and fine-tuning approaches. Essential for understanding modern few-shot learning used in drug discovery, personalized medicine, and rapid model adaptation with limited data.

###  **[basics-contrastive-learning.ipynb](notebooks/basics-contrastive-learning.ipynb)** ⭐ **CRITICAL for Modern AI**
Teaches contrastive learning, a revolutionary self-supervised paradigm that learns representations without labels by contrasting positive and negative pairs. Covers InfoNCE loss, the temperature parameter, cosine similarity, and data augmentation strategies. Implements a SimCLR-style model from scratch to demonstrate how models like CLIP, MoCo, and modern foundation models learn powerful representations from unlabeled data. Essential for understanding modern pre-training approaches.

###  **[basics-self-supervised-learning.ipynb](notebooks/basics-self-supervised-learning.ipynb)** ⭐ **CRITICAL for Modern AI**
Comprehensive overview of self-supervised learning - the paradigm powering modern foundation models. Covers five major approaches: Rotation Prediction (learning spatial features), Jigsaw Puzzles (understanding object parts), Masked Autoencoding (reconstructing missing regions), SimCLR (contrastive learning with augmented views), and BERT-style Masking (predicting masked words). Demonstrates how these methods generate "free" labels from data structure itself, enabling models to learn transferable representations without manual labeling. Essential for understanding modern pretraining approaches like GPT, BERT, CLIP, and MAE that have revolutionized AI. Requires understanding of autoencoders (TIER 12), transformers (TIER 10), and contrastive learning.

###  **[forward-forward.ipynb](notebooks/forward-forward.ipynb)**
Implements Geoffrey Hinton's Forward-Forward algorithm as an alternative to backpropagation, teaching layer-local learning through contrastive "goodness" functions with positive/negative samples, eliminating the need for backward gradient passes.

###  **[neuroevolution-tic-tac-toe.ipynb](notebooks/neuroevolution-tic-tac-toe.ipynb)**
Implements neuroevolution using genetic algorithms to train neural networks for tic-tac-toe, teaching evolutionary computation concepts (selection, mutation, fitness evaluation) as an alternative to gradient-based learning methods.

###  **[1bit-neural-networks.ipynb](notebooks/1bit-neural-networks.ipynb)**
Implements extreme quantization where weights are constrained to {-1, +1} (binary) or {-1, 0, +1} (ternary/1.58-bit). Demonstrates the straight-through estimator trick for training non-differentiable functions, achieving 8-32x memory reduction with minimal accuracy loss.

###  **[multi-task-learning.ipynb](notebooks/multi-task-learning.ipynb)** ⭐ **CRITICAL for Production Systems**
Comprehensive guide to multi-task learning (MTL), the paradigm of training a single model on multiple related tasks simultaneously. Teaches hard parameter sharing (shared encoder + task-specific heads), soft parameter sharing (separate networks with regularization), and cross-stitch networks (flexible layer-wise information exchange). Covers the critical challenge of loss balancing with naive summation, manual weighting, uncertainty-based weighting (automatic task weight learning), and gradient similarity analysis for detecting task relationships. Implements complete MTL systems for vision (CIFAR-10 with classification + coarse labels + color prediction) and toy regression tasks. Demonstrates when MTL helps (related tasks, limited data, auxiliary tasks) and how it provides implicit regularization preventing overfitting. Covers practical considerations including task sampling strategies (uniform, proportional, temperature-based). Essential for production systems doing multi-objective optimization, real-world applications combining related tasks (autonomous driving: detection + segmentation + depth), and maximizing data efficiency with limited labels per task.

###  **[curriculum-learning.ipynb](notebooks/curriculum-learning.ipynb)** ⭐ **Important Training Paradigm**
Comprehensive guide to curriculum learning, the paradigm of ordering training examples from easy to hard to accelerate learning and improve generalization. Teaches multiple difficulty metrics: image variance, edge density, loss-based scoring (high loss = hard), confidence-based (low confidence = hard), and domain knowledge heuristics. Covers curriculum strategies including baby steps (gradual difficulty increase), one-pass (single progression), spiral (revisit with increasing difficulty), and self-paced learning (model determines its own pace). Implements complete curriculum systems on CIFAR-10 with edge density scoring and arithmetic sequences with length-based difficulty. Demonstrates transfer curriculum (auxiliary task as stepping stone: grayscale → color), robustness to noisy labels (learn clean examples first), and anti-curriculum experiments (hard-first, controversial approach). Shows when curriculum helps: noisy data (identifying clean examples), hard optimization (avoiding local minima), domain shift (gradual adaptation), and sample efficiency (better generalization with limited data). Includes self-paced learning algorithm with adaptive thresholds and curriculum scheduler with configurable strategies. Essential for training on noisy datasets, improving sample efficiency, and understanding how training order affects model convergence and final performance.

###  **[active-learning.ipynb](notebooks/active-learning.ipynb)** ⭐ **CRITICAL for Data-Efficient Learning**
Comprehensive introduction to active learning, the paradigm where models intelligently select which data points to label, reducing annotation costs by 10-100x. Teaches the complete active learning loop (train → query → label → repeat), core query strategies including uncertainty sampling (least confidence, margin, entropy), query-by-committee (ensemble disagreement), and diversity sampling (K-means clustering). Implements all strategies from scratch on MNIST, demonstrating dramatic label efficiency improvements over random sampling. Covers deep active learning with MC Dropout for uncertainty estimation, batch mode active learning balancing uncertainty with diversity, and evaluation metrics (learning curves, area under learning curve, label efficiency). Addresses practical considerations including cold start problem (stratified initialization), stopping criteria (plateau detection), computational costs, and class imbalance handling. Shows real-world applications in medical imaging, autonomous driving, domain-specific NLP, and anomaly detection where annotation is expensive. Essential for production systems with limited annotation budgets, rare event detection, and maximizing performance with minimal labeling effort.

---

## TIER 16: Reinforcement Learning

**Learning through interaction with environments.**

###  **[rl-q-learning.ipynb](notebooks/rl-q-learning.ipynb)** ⭐ **FOUNDATIONAL RL**
Teaches foundational reinforcement learning through Q-learning on GridWorld and CartPole. Covers Markov Decision Processes (MDPs), value functions, Q-tables, the Bellman equation, temporal difference learning, and exploration vs exploitation (ε-greedy). Builds complete understanding of how agents learn from rewards through trial and error. Essential prerequisite for all modern RL algorithms (DQN, PPO, etc.) and policy-based methods.

###  **[mcts-simple.ipynb](notebooks/mcts-simple.ipynb)**
Introduces Monte Carlo Tree Search (MCTS) algorithm through a Tic-Tac-Toe implementation, explaining the four phases (selection, expansion, simulation, backpropagation) and UCB1 formula for balancing exploration vs exploitation.

###  **[rl-policy-gradients.ipynb](notebooks/rl-policy-gradients.ipynb)** ⭐ **CRITICAL for Modern RL**
Comprehensive introduction to policy gradient methods that directly optimize policies through gradient ascent. Covers the policy gradient theorem, REINFORCE algorithm, variance reduction with baselines, Actor-Critic methods combining policy and value learning, and PPO (Proximal Policy Optimization) basics. Demonstrates the progression from high-variance REINFORCE to state-of-the-art PPO on CartPole, teaching the foundations of modern reinforcement learning used in robotics, game playing, and autonomous systems.
###  **[rl-deep-q-networks.ipynb](notebooks/rl-deep-q-networks.ipynb)** ⭐ **CRITICAL - Bridge to Deep RL**
Bridges tabular Q-learning to deep reinforcement learning by introducing Deep Q-Networks (DQN). Explains why neural networks are needed for Q-value approximation in large state spaces, demonstrates the critical **experience replay** mechanism that breaks correlation in training data, and shows how **target networks** stabilize learning by preventing the moving target problem. Implements complete DQN from scratch on CartPole, with ablation studies proving each component's importance. Essential foundation for understanding all modern deep RL algorithms (PPO, SAC, etc.).

###  **[rl-advanced-policy-methods.ipynb](notebooks/rl-advanced-policy-methods.ipynb)** ⭐ **CRITICAL - State-of-the-Art RL**
Bridges basic policy gradients to state-of-the-art continuous control algorithms. Covers **A2C/A3C** (n-step returns and parallel environment collection), **SAC** (Soft Actor-Critic with maximum entropy RL for automatic exploration and sample efficiency), and **TD3** (Twin Delayed DDPG with twin critics and delayed updates). Implements all three algorithms from scratch on CartPole and Pendulum environments, demonstrating continuous action spaces, off-policy learning, and modern techniques used in robotics. Essential for understanding modern RL methods used in real-world applications.

###  **[rlhf-alignment.ipynb](notebooks/rlhf-alignment.ipynb)** ⭐ **CRITICAL - The Technique Behind ChatGPT**
Comprehensive guide to Reinforcement Learning from Human Feedback (RLHF) - the technique that transformed ChatGPT from a text predictor into a helpful assistant. Teaches the complete three-stage pipeline: **SFT** (supervised fine-tuning on demonstrations), **Reward Modeling** (learning to predict human preferences using the Bradley-Terry model), and **PPO optimization** (maximizing reward while avoiding reward hacking via KL penalty). Implements reward model training from preference pairs, PPO with KL divergence constraint, and **DPO** (Direct Preference Optimization) as a simpler alternative. Covers Constitutional AI (self-critique for reduced human supervision), practical challenges (reward hacking, distribution shift, evaluation), and real-world applications. Demonstrates sentiment steering on GPT-2 to show before/after alignment effects. Essential for understanding how modern AI assistants (ChatGPT, Claude, Llama 2) learn to align with human values and preferences. Requires understanding of PPO, transformers, and preference learning.

###  **[rl-model-based.ipynb](notebooks/rl-model-based.ipynb)** ⭐ **Alternative RL Paradigm**
Introduces model-based reinforcement learning as an alternative to model-free approaches. Teaches how to learn a **world model** that predicts environment dynamics (transitions and rewards), then use it for planning to dramatically improve sample efficiency. Covers the **Dyna algorithm** (combining real and simulated experiences), **prioritized sweeping** (smart planning), and full planning with value iteration. Demonstrates the fundamental trade-off between computational cost and sample efficiency, showing why model-based RL is critical when real environment interactions are expensive (robotics, clinical trials, etc.). Essential for understanding modern algorithms like MuZero and Dreamer.

###  **[wip-rl-world-model-01-repr.ipynb](notebooks/wip-rl-world-model-01-repr.ipynb)** *(Work in Progress)*
Part 1 of a world model series teaching how to compress gameplay frames into low-dimensional latent representations using convolutional autoencoders for the Tetris GameBoy environment.

###  **[wip-rl-world-model-02-dynamics.ipynb](notebooks/wip-rl-world-model-02-dynamics.ipynb)** *(Work in Progress)*
Part 2 of a world model series teaching how to train a dynamics model to predict future latent states in a reinforcement learning environment given actions in the Tetris GameBoy environment.

###  **[wip-rl-atari-pong-imitation.ipynb](notebooks/wip-rl-atari-pong-imitation.ipynb)** *(Work in Progress)*
Focuses on reinforcement learning for Atari Pong using imitation learning techniques to train agents by learning from expert demonstrations.

---

## TIER 17: Fun & Specialized Applications

**Unique and interesting applications.**

###  **[chip8-emulator.ipynb](notebooks/chip8-emulator.ipynb)**
Builds a complete CHIP-8 emulator from scratch to teach fundamental emulation concepts including the fetch-decode-execute cycle, memory management, opcodes, and how computers simulate other computers through software.

###  **[wip-debate-generator.ipynb](notebooks/wip-debate-generator.ipynb)** *(Work in Progress)*
Teaches how to build an AI-powered political debate simulator using LangChain and LLMs, creating multiple AI agents representing different political parties that engage in structured debates with text-to-speech and video avatar generation.

---

## 🎓 Learning Path Recommendations

### **Quick Start Path (Core Essentials)**
If you're short on time, focus on notebooks marked with ⭐:
1. Complete all of Tier 1-4 (foundations)
2. z2h-01-backprop.ipynb (CRITICAL)
3. basics-kl-divergence.ipynb (for VAEs/RL)
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

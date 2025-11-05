# AI/ML Notebooks - Table of Contents

## 🎯 PURPOSE & MAINTENANCE REQUIREMENT

**This file MUST be maintained to delineate the optimal learning order for a complete AI/ML n00b to make it to god-tier by following the specified sequence.**

### Critical Constraint:
When this TOC is updated (e.g., when new notebooks are added or existing notebooks are modified), this constraint MUST ALWAYS be upheld - even if it means restructuring the existing order. The learning progression from foundational to advanced topics is the PRIMARY organizing principle.

### Maintenance Rules:
1. **New notebooks** must be inserted in their appropriate tier based on conceptual prerequisites
2. **Updated notebooks** may require repositioning if their content changes significantly
3. **Tier structure** should reflect clear learning progression with minimal prerequisite violations
4. **Dependencies** between notebooks should be explicitly considered (e.g., VAEs require understanding KL divergence)

---

## 📚 Learning Path Overview

This repository contains educational Jupyter notebooks covering fundamental to advanced AI/ML concepts. Each notebook is designed to build deep intuitions through hands-on implementation and progressive learning.

**This TOC is organized in optimal learning order** - starting from absolute foundations and progressively building to advanced topics. Follow this sequence for the most effective learning path from n00b to god-tier.

---

## TIER 1: Absolute Foundations

**Start here!** These are the bare essentials before anything else.

### **basics-numpy-fundamentals.ipynb**
Foundation of numerical computing in Python. Covers array creation, indexing, slicing, broadcasting rules (critical for tensors!), universal functions, and random number generation. Essential prerequisite for understanding PyTorch tensors and all subsequent ML concepts.

### **basics-tensors-operations.ipynb**
Introduction to tensor operations and manipulations in PyTorch, covering shapes, broadcasting, indexing, and fundamental tensor algebra that forms the foundation of deep learning computations.

### **basics-tensor-multiplication.ipynb**
Deep dive into matrix multiplication, broadcasting rules, and efficient tensor operations. Builds intuition for how neural network layers actually perform computations through matrix operations.

### **basics-linear-regression.ipynb**
Builds linear regression from scratch to develop deep intuitions about gradient descent, loss functions, and optimization. Compares iterative gradient descent with the analytical closed-form solution, teaching fundamental machine learning concepts that underpin all neural network training.

### **basics-metrics.ipynb**
Comprehensive guide to evaluation metrics (accuracy, precision, recall, F1, ROC-AUC, etc.) for classification and regression tasks, teaching when to use each metric and how to interpret them in different contexts.

### **basics-probability-distributions.ipynb**
Foundational introduction to probability distributions that appear throughout machine learning. Covers discrete distributions (PMF: Bernoulli, Categorical, Binomial) and continuous distributions (PDF: Uniform, Gaussian/Normal), teaching mean/variance/sampling with visual intuitions. Explains the Central Limit Theorem and connects distributions to ML concepts like cross-entropy loss, VAEs, and model outputs. Essential for understanding loss functions, KL divergence, and generative models.

### **basics-loss-functions.ipynb**
Provides comprehensive coverage of loss functions (MSE, MAE, cross-entropy, hinge loss) used in machine learning, explaining their mathematical properties and when to use each. Visualizes how different losses handle outliers, class imbalance, and various prediction tasks through interactive experiments.

### **basics-activation-functions.ipynb**
Teaches the fundamental activation functions used in neural networks (ReLU, sigmoid, tanh, etc.) and their properties. Explores how different activation functions affect gradient flow, training dynamics, and model expressiveness through visualizations and hands-on comparisons.

### **logistic-regression.ipynb**
Teaches how to build a binary linear classifier using logistic regression on 2D clustered data, covering gradient descent, loss functions (binary cross-entropy), and decision boundary visualization.

### **basics-softmax-multiclass.ipynb**
Bridges from binary to multiclass classification by teaching the softmax function, one-hot encoding, and categorical cross-entropy loss. Builds a complete multiclass classifier from scratch on the Iris dataset, demonstrating how softmax generalizes sigmoid and why it's the standard approach for multiclass problems. Essential foundation before neural networks.

---

## TIER 2: Core ML Principles

**Understanding how to build models that generalize.**

### **basics-overfitting-underfitting.ipynb**
Explores the fundamental machine learning challenge of balancing model capacity using polynomial regression as a teaching tool. Demonstrates how to detect and address underfitting (high bias) and overfitting (high variance) through learning curves, validation strategies, and the bias-variance tradeoff.

### **basics-bias-variance-tradeoff.ipynb**
Theoretical exploration of the bias-variance decomposition, showing how model error can be broken down into irreducible error, bias, and variance components to guide model selection and complexity decisions.

### **basics-train-test-val-split.ipynb**
Foundational explanation of why we split data into train/validation/test sets, teaching the philosophy behind each split and when to use them. Covers data leakage prevention (preprocessing order, target leakage), temporal split considerations for time-series data, and common mistakes that lead to overly optimistic performance estimates. Essential prerequisite for understanding cross-validation.

### **basics-cross-validation.ipynb**
Explores three essential cross-validation techniques: K-Fold, Stratified K-Fold, and Time Series Split. Teaches when to use each method, how they provide more reliable performance estimates than single train/test splits, and how to avoid data leakage in temporal data.

### **basics-regularization.ipynb**
Explores regularization techniques (L1, L2, dropout, early stopping) that prevent overfitting by constraining model complexity. Shows how regularization adds inductive bias to help models generalize better to unseen data.

### **basics-confidence-intervals.ipynb**
Teaches how to quantify uncertainty in model predictions and evaluation metrics using statistical confidence intervals. Demonstrates bootstrap methods and hypothesis testing to understand whether performance differences between models are statistically significant.

---

## TIER 3: Data & Optimization

**Making the most of your data and training.**

### **basics-data-normalization.ipynb** ⭐
Deep dive into data normalization/standardization - min-max scaling vs standardization (z-score), when to use each, why neural networks need normalized inputs, and critically: fitting on train data then applying to test to avoid data leakage. Essential preprocessing skill that directly impacts model convergence and performance.

### **basics-feature-engineering.ipynb**
Covers techniques for creating informative features from raw data including encoding categorical variables, handling missing values, scaling, and domain-specific feature extraction to improve model performance.

### **basics-data-augmentation.ipynb**
Teaches data augmentation strategies for artificially expanding training datasets through transformations (rotations, flips, crops for images; synonym replacement for text) to improve model generalization and robustness.

### **basics-mini-batch-gradient-descent.ipynb** ⭐ **CRITICAL**
Teaches the three flavors of gradient descent (full-batch, mini-batch, stochastic) and essential terminology (epochs, steps, iterations, batches). Explains why mini-batching works through gradient noise analysis, memory/computational trade-offs, and practical batch size selection. Critical prerequisite for understanding optimizers, as all modern optimizers assume mini-batch training with noisy gradients.

### **basics-optimizers.ipynb**
Deep dive into optimization algorithms (SGD, Momentum, RMSprop, Adam) that train neural networks, building intuition through visualization of their paths through loss landscapes. Explains how each optimizer addresses specific challenges like ravines, saddle points, and different parameter scales, with practical guidance on when to use each.

### **basics-learning-rate-schedules.ipynb**
Teaches how to improve training by dynamically adjusting learning rates over time using schedules like step decay, exponential decay, and cosine annealing. Demonstrates why starting with high learning rates and gradually reducing them leads to better convergence, with warmup techniques for large models.

---

## TIER 4: Deep Learning Foundations

**Now we can start with neural networks!**

### **basics-computational-graphs.ipynb** ⭐ **CRITICAL**
Teaches how to represent any computation as a directed acyclic graph (DAG) where nodes are values/operations and edges represent data flow. Builds visual intuition for how graphs naturally encode the chain rule, making backpropagation obvious. Shows how PyTorch's automatic differentiation builds and traverses these graphs during forward and backward passes. **Essential foundation before backprop!**

### **basics-neural-network-fundamentals.ipynb**
Foundation for understanding what neural networks actually are. Covers the core building blocks: single neurons (perceptrons), layers as collections of neurons, and networks as stacked layers. Explains forward pass computation, architecture terminology (width, depth, parameters), why we stack layers for hierarchical feature learning, and provides intuition for the Universal Approximation Theorem. Essential prerequisite for understanding backpropagation and training.

### **z2h-01-backprop.ipynb** ⭐ **CRITICAL**
Teaches backpropagation from scratch by building a micrograd-style autograd engine that tracks computational graphs, calculates gradients using the chain rule, and trains neural networks using gradient descent. **Must understand this deeply before proceeding!**

### **basics-gradient-stability.ipynb**
Explores the vanishing and exploding gradient problems that plague deep neural networks, demonstrating how gradients can shrink or grow exponentially during backpropagation through many layers and why this matters for training.

### **basics-gradient-clipping.ipynb**
Teaches gradient clipping as a solution to exploding gradients, showing how to cap gradient magnitudes to stabilize training in recurrent networks and deep architectures.

### **basics-gradient-normalization.ipynb**
Covers gradient normalization techniques that rescale gradients to have consistent magnitude across training, improving optimization stability especially in networks with varying layer depths.

### **basics-batch-normalization.ipynb**
Explains batch normalization, a technique that normalizes layer inputs to stabilize and accelerate training. Covers the internal covariate shift problem, demonstrates how BatchNorm maintains stable distributions across layers, and shows its impact on training speed and performance.

### **basics-layer-normalization.ipynb**
Introduces layer normalization as an alternative to batch normalization that normalizes across features rather than batch dimensions, making it more suitable for recurrent networks and transformers.

---

## TIER 5: First Neural Networks

**Simple neural networks for language modeling.**

### **z2h-02-bigram-lm.ipynb**
Introduces character-level bigram language models by manually building probability distributions from character pair frequencies, then recreating the same model using PyTorch with backpropagation and gradient descent.

### **z2h-03-mlp-lm.ipynb**
Builds a multi-layer perceptron (MLP) language model for character-level text generation, showing how adding hidden layers and nonlinearities enables the model to capture more complex patterns than simple bigrams.

### **z2h-04-optimization-pt1.ipynb**
Focuses on neural network optimization techniques including weight initialization strategies (Xavier, Kaiming), analyzing activation and gradient flow through layers, and understanding how these choices critically affect training performance.

### **z2h-05-optimization-pt2-wip.ipynb** *(Work in Progress)*
Continues optimization topics by migrating code to PyTorch and conducting experiments with initialization strategies and batch normalization to understand their impact on data and gradient flow through neural networks.

### **z2h-06-backprop-ninja-wip.ipynb** *(Work in Progress)*
Advanced backpropagation practice where students manually implement backward passes for complex operations (cross-entropy, batch normalization) to deeply understand gradient computation in neural networks.

---

## TIER 6: Convolutional Neural Networks

**Computer vision fundamentals.**

### **basics-tensors-convolution.ipynb**
Teaches convolutional operations from first principles, showing how filters slide across inputs to extract features. Essential for understanding CNNs and image processing architectures.

### **classification-image.ipynb**
Demonstrates building image classifiers with configurable architectures (CNN or MLP) on multiple datasets (CIFAR-10, MNIST, Fashion-MNIST), using PyTorch Lightning for training with automatic checkpointing and W&B logging.

---

## TIER 7: Information Theory & Embeddings

**Important concepts for advanced topics.**

### **basics-kl-divergence.ipynb** ⭐ **CRITICAL for VAEs & RL**
Teaches Kullback-Leibler (KL) divergence as a fundamental information-theoretic measure of how one probability distribution differs from another. Builds intuition from first principles through information content, entropy, and cross-entropy, explaining the asymmetry of KL divergence, forward vs reverse KL behavior (mode-seeking vs mode-covering), and critical applications in VAEs, reinforcement learning policy optimization (TRPO/PPO), variational inference, and distribution matching.

### **basics-embeddings.ipynb**
Introduces the concept of learned dense vector representations for discrete objects (words, items, users), showing how embeddings capture semantic relationships and enable neural networks to process categorical data effectively.

---

## TIER 8: Unsupervised Learning Basics

**Learning without labels.**

### **kmeans-from-scratch.ipynb**
Builds the k-means clustering algorithm from first principles, showing how it partitions data through iterative assignment and update steps. Covers distance metrics, initialization strategies (k-means++), convergence guarantees, and the elbow method for choosing k.

---

## TIER 9: Recurrent Neural Networks

**Sequential data and memory.**

### **rnn-from-scratch.ipynb**
A complete implementation of Recurrent Neural Networks (RNNs) for character-level language modeling. Teaches how networks maintain "memory" through hidden states, the mechanics of backpropagation through time (BPTT), and why RNNs suffer from vanishing gradients.

### **lstm-from-scratch.ipynb**
Builds LSTM networks from scratch using only NumPy to develop deep understanding of gate mechanisms (forget, input, output), cell state vs hidden state, and how LSTMs solve the vanishing gradient problem for long-range dependencies.

### **wip-bit-parity-rnn.ipynb** *(Work in Progress)*
Teaches the fundamentals of recurrent neural networks (RNNs) through the bit-parity classification task, demonstrating building an RNN from scratch to classify binary sequences as having odd or even parity.

### **wip-bit-parity-gru.ipynb** *(Work in Progress)*
Teaches the fundamentals of Gated Recurrent Units (GRUs) through the bit-parity classification task, implementing a GRU cell from scratch with explicit update gates, reset gates, and candidate hidden states.

---

## TIER 10: Attention & Transformers

**Modern sequence modeling - the revolution.**

### **basics-attention-mechanism.ipynb**
Introduces the attention mechanism that revolutionized deep learning, explaining how models can "focus" on relevant parts of input sequences. Builds intuition through step-by-step implementation of scaled dot-product attention and demonstrates its use in sequence-to-sequence tasks.

### **word2vec-from-scratch.ipynb**
Implements Word2Vec using the skip-gram model with negative sampling to learn word embeddings. Demonstrates the distributional hypothesis ("you shall know a word by the company it keeps") and shows how vector arithmetic captures semantic relationships.

### **transformer-from-scratch.ipynb** ⭐ **CRITICAL for Modern AI**
Builds a Transformer architecture incrementally from basic building blocks to a complete model. Covers the self-attention mechanism (the core innovation), multi-head attention, positional encodings, and how Transformers process sequences in parallel unlike RNNs.

### **z2h-07-wavenet-lm-wip.ipynb** *(Work in Progress)*
Implements a WaveNet-inspired hierarchical language model using custom layers (embeddings, batch normalization, sequential flattening) to process longer character contexts more efficiently than flat architectures.

---

## TIER 11: Text Applications

**Practical NLP tasks end-to-end.**

### **classification-text.ipynb**
Explores sentiment analysis on IMDb movie reviews using three progressive approaches: TF-IDF + Logistic Regression (baseline), LSTM neural networks, and fine-tuned BERT, comparing their performance and trade-offs.

### **generation-text.ipynb**
Comprehensively covers text generation from bigram statistical models through neural approaches (MLP, RNN, LSTM) to transformers, including sampling strategies (temperature, top-k, nucleus) and latent space manipulation.

### **reconstruction-text.ipynb**
Teaches sequence-to-sequence autoencoders for text reconstruction using encoder-decoder architectures, demonstrating latent space interpolation, teacher forcing training strategies, and applications like semantic similarity measurement.

### **denoising-text.ipynb**
Demonstrates text correction and denoising techniques using seq2seq models with attention and transformers, including noise injection strategies, edit distance baselines, and evaluation metrics (CER/WER).

---

## TIER 12: Generative Models for Images

**Learning to generate and reconstruct images.**

### **reconstruction-image.ipynb**
Explores image reconstruction through three autoencoder variants (Vanilla, VAE, VQ-VAE) on MNIST, demonstrating compression, latent space analysis, and applications like denoising and anomaly detection.

### **generation-image.ipynb** ⭐ **VAEs & GANs**
Teaches image generation using Variational Autoencoders (VAE) and Deep Convolutional GANs (DCGAN) on MNIST, covering latent space exploration, interpolation, and the trade-offs between explicit vs implicit generative models.

### **denoising-image.ipynb**
Focuses on removing noise from corrupted images using convolutional autoencoders and denoising autoencoders (DAE), teaching how models learn robust representations by reconstructing clean data from noisy inputs.

---

## TIER 13: Advanced Architectures

**Beyond standard feedforward, convolutional, and recurrent networks.**

### **graph-neural-networks.ipynb**
Introduces Graph Neural Networks (GNNs) for learning on graph-structured data like social networks and molecules. Teaches the message passing framework where nodes aggregate information from neighbors, demonstrated through semi-supervised node classification.

### **energy-based-neural-networks.ipynb**
Explores energy-based models (EBMs) including Hopfield Networks and Restricted Boltzmann Machines (RBMs). Teaches how energy functions assign low energy to "good" configurations, the Gibbs distribution, and contrastive divergence for training.

---

## TIER 14: Fascinating Phenomena

**Interesting research findings that challenge intuitions.**

### **double-descent.ipynb**
Explores the double descent phenomenon where test error decreases again after the interpolation threshold, challenging classical bias-variance tradeoff intuitions and demonstrating why overparameterized models can generalize better than expected.

### **grokking.ipynb**
Demonstrates the grokking phenomenon where neural networks suddenly transition from memorization to generalization after prolonged training on algorithmic tasks (modular arithmetic), showing how models can achieve perfect generalization thousands of epochs after overfitting.

---

## TIER 15: Alternative Training Paradigms

**Beyond standard backpropagation and gradient descent.**

### **forward-forward.ipynb**
Implements Geoffrey Hinton's Forward-Forward algorithm as an alternative to backpropagation, teaching layer-local learning through contrastive "goodness" functions with positive/negative samples, eliminating the need for backward gradient passes.

### **neuroevolution-tic-tac-toe.ipynb**
Implements neuroevolution using genetic algorithms to train neural networks for tic-tac-toe, teaching evolutionary computation concepts (selection, mutation, fitness evaluation) as an alternative to gradient-based learning methods.

### **1bit-neural-networks.ipynb**
Implements extreme quantization where weights are constrained to {-1, +1} (binary) or {-1, 0, +1} (ternary/1.58-bit). Demonstrates the straight-through estimator trick for training non-differentiable functions, achieving 8-32x memory reduction with minimal accuracy loss.

---

## TIER 16: Reinforcement Learning

**Learning through interaction with environments.**

### **mcts-simple.ipynb**
Introduces Monte Carlo Tree Search (MCTS) algorithm through a Tic-Tac-Toe implementation, explaining the four phases (selection, expansion, simulation, backpropagation) and UCB1 formula for balancing exploration vs exploitation.

### **wip-rl-world-model-01-repr.ipynb** *(Work in Progress)*
Part 1 of a world model series teaching how to compress gameplay frames into low-dimensional latent representations using convolutional autoencoders for the Tetris GameBoy environment.

### **wip-rl-world-model-02-dynamics.ipynb** *(Work in Progress)*
Part 2 of a world model series teaching how to train a dynamics model to predict future latent states in a reinforcement learning environment given actions in the Tetris GameBoy environment.

### **wip-rl-atari-pong-imitation.ipynb** *(Work in Progress)*
Focuses on reinforcement learning for Atari Pong using imitation learning techniques to train agents by learning from expert demonstrations.

---

## TIER 17: Fun & Specialized Applications

**Unique and interesting applications.**

### **chip8-emulator.ipynb**
Builds a complete CHIP-8 emulator from scratch to teach fundamental emulation concepts including the fetch-decode-execute cycle, memory management, opcodes, and how computers simulate other computers through software.

### **wip-debate-generator.ipynb** *(Work in Progress)*
Teaches how to build an AI-powered political debate simulator using LangChain and LLMs, creating multiple AI agents representing different political parties that engage in structured debates with text-to-speech and video avatar generation.

## Learning Path Recommendations

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
- Tiers 1-4 → Tier 7 (esp. KL divergence) → Tier 16

**Research/Theory Track:**
- Tiers 1-4 → Tier 7 → Tier 13 → Tier 14 → Tier 15

---

**Note:** This learning progression builds conceptual dependencies systematically. Each tier assumes mastery of previous tiers. Notebooks marked as "Work in Progress" may be incomplete but are placed where they fit conceptually in the learning path.

# AI/ML Notebooks - Table of Contents

This repository contains educational Jupyter notebooks covering fundamental to advanced AI/ML concepts. Each notebook is designed to build deep intuitions through hands-on implementation and progressive learning.

---

## Basics Series

Core ML/DL concepts and fundamental techniques that underpin all modern machine learning.

### **basics-linear-regression.ipynb**
Builds linear regression from scratch to develop deep intuitions about gradient descent, loss functions, and optimization. Compares iterative gradient descent with the analytical closed-form solution, teaching fundamental machine learning concepts that underpin all neural network training.

### **basics-tensors-operations.ipynb**
Introduction to tensor operations and manipulations in PyTorch, covering shapes, broadcasting, indexing, and fundamental tensor algebra that forms the foundation of deep learning computations.

### **basics-tensor-multiplication.ipynb**
Deep dive into matrix multiplication, broadcasting rules, and efficient tensor operations. Builds intuition for how neural network layers actually perform computations through matrix operations.

### **basics-tensors-convolution.ipynb**
Teaches convolutional operations from first principles, showing how filters slide across inputs to extract features. Essential for understanding CNNs and image processing architectures.

### **basics-activation-functions.ipynb**
Teaches the fundamental activation functions used in neural networks (ReLU, sigmoid, tanh, etc.) and their properties. Explores how different activation functions affect gradient flow, training dynamics, and model expressiveness through visualizations and hands-on comparisons.

### **basics-loss-functions.ipynb**
Provides comprehensive coverage of loss functions (MSE, MAE, cross-entropy, hinge loss) used in machine learning, explaining their mathematical properties and when to use each. Visualizes how different losses handle outliers, class imbalance, and various prediction tasks through interactive experiments.

### **basics-kl-divergence.ipynb**
Teaches Kullback-Leibler (KL) divergence as a fundamental information-theoretic measure of how one probability distribution differs from another. Builds intuition from first principles through information content, entropy, and cross-entropy, explaining the asymmetry of KL divergence, forward vs reverse KL behavior (mode-seeking vs mode-covering), and critical applications in VAEs, reinforcement learning policy optimization (TRPO/PPO), variational inference, and distribution matching.

### **basics-optimizers.ipynb**
Deep dive into optimization algorithms (SGD, Momentum, RMSprop, Adam) that train neural networks, building intuition through visualization of their paths through loss landscapes. Explains how each optimizer addresses specific challenges like ravines, saddle points, and different parameter scales, with practical guidance on when to use each.

### **basics-learning-rate-schedules.ipynb**
Teaches how to improve training by dynamically adjusting learning rates over time using schedules like step decay, exponential decay, and cosine annealing. Demonstrates why starting with high learning rates and gradually reducing them leads to better convergence, with warmup techniques for large models.

### **basics-regularization.ipynb**
Explores regularization techniques (L1, L2, dropout, early stopping) that prevent overfitting by constraining model complexity. Shows how regularization adds inductive bias to help models generalize better to unseen data.

### **basics-overfitting-underfitting.ipynb**
Explores the fundamental machine learning challenge of balancing model capacity using polynomial regression as a teaching tool. Demonstrates how to detect and address underfitting (high bias) and overfitting (high variance) through learning curves, validation strategies, and the bias-variance tradeoff.

### **basics-bias-variance-tradeoff.ipynb**
Theoretical exploration of the bias-variance decomposition, showing how model error can be broken down into irreducible error, bias, and variance components to guide model selection and complexity decisions.

### **basics-cross-validation.ipynb**
Explores three essential cross-validation techniques: K-Fold, Stratified K-Fold, and Time Series Split. Teaches when to use each method, how they provide more reliable performance estimates than single train/test splits, and how to avoid data leakage in temporal data.

### **basics-metrics.ipynb**
Comprehensive guide to evaluation metrics (accuracy, precision, recall, F1, ROC-AUC, etc.) for classification and regression tasks, teaching when to use each metric and how to interpret them in different contexts.

### **basics-confidence-intervals.ipynb**
Teaches how to quantify uncertainty in model predictions and evaluation metrics using statistical confidence intervals. Demonstrates bootstrap methods and hypothesis testing to understand whether performance differences between models are statistically significant.

### **basics-feature-engineering.ipynb**
Covers techniques for creating informative features from raw data including encoding categorical variables, handling missing values, scaling, and domain-specific feature extraction to improve model performance.

### **basics-data-augmentation.ipynb**
Teaches data augmentation strategies for artificially expanding training datasets through transformations (rotations, flips, crops for images; synonym replacement for text) to improve model generalization and robustness.

### **basics-embeddings.ipynb**
Introduces the concept of learned dense vector representations for discrete objects (words, items, users), showing how embeddings capture semantic relationships and enable neural networks to process categorical data effectively.

### **basics-attention-mechanism.ipynb**
Introduces the attention mechanism that revolutionized deep learning, explaining how models can "focus" on relevant parts of input sequences. Builds intuition through step-by-step implementation of scaled dot-product attention and demonstrates its use in sequence-to-sequence tasks.

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

## Zero to Hero Series

A progressive course series building from fundamental concepts to advanced neural language models.

### **z2h-01-backprop.ipynb**
Teaches backpropagation from scratch by building a micrograd-style autograd engine that tracks computational graphs, calculates gradients using the chain rule, and trains neural networks using gradient descent.

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

### **z2h-07-wavenet-lm-wip.ipynb** *(Work in Progress)*
Implements a WaveNet-inspired hierarchical language model using custom layers (embeddings, batch normalization, sequential flattening) to process longer character contexts more efficiently than flat architectures.

---

## From Scratch Implementations

Building neural architectures from first principles to understand their inner workings.

### **rnn-from-scratch.ipynb**
A complete implementation of Recurrent Neural Networks (RNNs) for character-level language modeling. Teaches how networks maintain "memory" through hidden states, the mechanics of backpropagation through time (BPTT), and why RNNs suffer from vanishing gradients.

### **lstm-from-scratch.ipynb**
Builds LSTM networks from scratch using only NumPy to develop deep understanding of gate mechanisms (forget, input, output), cell state vs hidden state, and how LSTMs solve the vanishing gradient problem for long-range dependencies.

### **transformer-from-scratch.ipynb**
Builds a Transformer architecture incrementally from basic building blocks to a complete model. Covers the self-attention mechanism (the core innovation), multi-head attention, positional encodings, and how Transformers process sequences in parallel unlike RNNs.

### **word2vec-from-scratch.ipynb**
Implements Word2Vec using the skip-gram model with negative sampling to learn word embeddings. Demonstrates the distributional hypothesis ("you shall know a word by the company it keeps") and shows how vector arithmetic captures semantic relationships.

### **kmeans-from-scratch.ipynb**
Builds the k-means clustering algorithm from first principles, showing how it partitions data through iterative assignment and update steps. Covers distance metrics, initialization strategies (k-means++), convergence guarantees, and the elbow method for choosing k.

---

## Core ML Tasks

Practical implementations of fundamental machine learning tasks across modalities.

### **logistic-regression.ipynb**
Teaches how to build a binary linear classifier using logistic regression on 2D clustered data, covering gradient descent, loss functions (binary cross-entropy), and decision boundary visualization.

### **classification-image.ipynb**
Demonstrates building image classifiers with configurable architectures (CNN or MLP) on multiple datasets (CIFAR-10, MNIST, Fashion-MNIST), using PyTorch Lightning for training with automatic checkpointing and W&B logging.

### **classification-text.ipynb**
Explores sentiment analysis on IMDb movie reviews using three progressive approaches: TF-IDF + Logistic Regression (baseline), LSTM neural networks, and fine-tuned BERT, comparing their performance and trade-offs.

### **generation-image.ipynb**
Teaches image generation using Variational Autoencoders (VAE) and Deep Convolutional GANs (DCGAN) on MNIST, covering latent space exploration, interpolation, and the trade-offs between explicit vs implicit generative models.

### **generation-text.ipynb**
Comprehensively covers text generation from bigram statistical models through neural approaches (MLP, RNN, LSTM) to transformers, including sampling strategies (temperature, top-k, nucleus) and latent space manipulation.

### **reconstruction-image.ipynb**
Explores image reconstruction through three autoencoder variants (Vanilla, VAE, VQ-VAE) on MNIST, demonstrating compression, latent space analysis, and applications like denoising and anomaly detection.

### **reconstruction-text.ipynb**
Teaches sequence-to-sequence autoencoders for text reconstruction using encoder-decoder architectures, demonstrating latent space interpolation, teacher forcing training strategies, and applications like semantic similarity measurement.

### **denoising-image.ipynb**
Focuses on removing noise from corrupted images using convolutional autoencoders and denoising autoencoders (DAE), teaching how models learn robust representations by reconstructing clean data from noisy inputs.

### **denoising-text.ipynb**
Demonstrates text correction and denoising techniques using seq2seq models with attention and transformers, including noise injection strategies, edit distance baselines, and evaluation metrics (CER/WER).

---

## Advanced Topics & Research

Cutting-edge techniques and interesting phenomena in deep learning.

### **double-descent.ipynb**
Explores the double descent phenomenon where test error decreases again after the interpolation threshold, challenging classical bias-variance tradeoff intuitions and demonstrating why overparameterized models can generalize better than expected.

### **grokking.ipynb**
Demonstrates the grokking phenomenon where neural networks suddenly transition from memorization to generalization after prolonged training on algorithmic tasks (modular arithmetic), showing how models can achieve perfect generalization thousands of epochs after overfitting.

### **forward-forward.ipynb**
Implements Geoffrey Hinton's Forward-Forward algorithm as an alternative to backpropagation, teaching layer-local learning through contrastive "goodness" functions with positive/negative samples, eliminating the need for backward gradient passes.

### **1bit-neural-networks.ipynb**
Implements extreme quantization where weights are constrained to {-1, +1} (binary) or {-1, 0, +1} (ternary/1.58-bit). Demonstrates the straight-through estimator trick for training non-differentiable functions, achieving 8-32x memory reduction with minimal accuracy loss.

### **energy-based-neural-networks.ipynb**
Explores energy-based models (EBMs) including Hopfield Networks and Restricted Boltzmann Machines (RBMs). Teaches how energy functions assign low energy to "good" configurations, the Gibbs distribution, and contrastive divergence for training.

### **graph-neural-networks.ipynb**
Introduces Graph Neural Networks (GNNs) for learning on graph-structured data like social networks and molecules. Teaches the message passing framework where nodes aggregate information from neighbors, demonstrated through semi-supervised node classification.

---

## Specialized Applications

Unique applications demonstrating AI/ML in diverse domains.

### **neuroevolution-tic-tac-toe.ipynb**
Implements neuroevolution using genetic algorithms to train neural networks for tic-tac-toe, teaching evolutionary computation concepts (selection, mutation, fitness evaluation) as an alternative to gradient-based learning methods.

### **mcts-simple.ipynb**
Introduces Monte Carlo Tree Search (MCTS) algorithm through a Tic-Tac-Toe implementation, explaining the four phases (selection, expansion, simulation, backpropagation) and UCB1 formula for balancing exploration vs exploitation.

### **chip8-emulator.ipynb**
Builds a complete CHIP-8 emulator from scratch to teach fundamental emulation concepts including the fetch-decode-execute cycle, memory management, opcodes, and how computers simulate other computers through software.

---

## Work in Progress

Notebooks currently under development.

### **wip-bit-parity-rnn.ipynb**
Teaches the fundamentals of recurrent neural networks (RNNs) through the bit-parity classification task, demonstrating building an RNN from scratch to classify binary sequences as having odd or even parity.

### **wip-bit-parity-gru.ipynb**
Teaches the fundamentals of Gated Recurrent Units (GRUs) through the bit-parity classification task, implementing a GRU cell from scratch with explicit update gates, reset gates, and candidate hidden states.

### **wip-debate-generator.ipynb**
Teaches how to build an AI-powered political debate simulator using LangChain and LLMs, creating multiple AI agents representing different political parties that engage in structured debates with text-to-speech and video avatar generation.

### **wip-rl-world-model-01-repr.ipynb**
Part 1 of a world model series teaching how to compress gameplay frames into low-dimensional latent representations using convolutional autoencoders for the Tetris GameBoy environment.

### **wip-rl-world-model-02-dynamics.ipynb**
Part 2 of a world model series teaching how to train a dynamics model to predict future latent states in a reinforcement learning environment given actions in the Tetris GameBoy environment.

### **wip-rl-atari-pong-imitation.ipynb**
Focuses on reinforcement learning for Atari Pong using imitation learning techniques to train agents by learning from expert demonstrations.

---

## Summary Statistics

- **Total notebooks:** 60
- **Basics series:** 22 notebooks covering foundational concepts
- **Zero to Hero series:** 7 notebooks (progressive course)
- **From Scratch implementations:** 5 notebooks
- **Core ML tasks:** 9 notebooks
- **Advanced topics:** 6 notebooks
- **Specialized applications:** 3 notebooks
- **Work in progress:** 6 notebooks

---

**Note:** This table of contents provides descriptions of what each notebook teaches. For optimal learning progression from beginner to advanced, notebooks should be ordered based on prerequisite knowledge and conceptual dependencies.

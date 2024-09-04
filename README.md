# aiml-notebooks

This repository contains my AI/ML experiments. Each notebook either recreates lessons I've learned or explores concepts I'm studying. I add more detail than the original lessons to solidify my understanding and create a reference for future review. My process is simple: complete a lesson, write a notebook about it, then iterate on it, cleaning up and expanding the content until I've clarified everything I don't fully understand. As a result, these notebooks often are more verbose than the original material (for better or worse).

## Neural Networks: Zero to Hero

These notebooks are detailed recreations of lessons from [Andrej Karpathy](https://karpathy.ai/)'s [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) course. 

1. **[Backprop from Scratch](https://colab.research.google.com/github/tsilva/aiml-notebooks/blob/main/karpathy-zero-to-hero/001-backprop-from-scratch.ipynb)**: This notebook introduces the foundational principles of **neural networks**, **backpropagation**, and **gradient descent**. It walks through constructing calculation graphs and using them to track calculation gradients, manually implementing backpropagation and gradient descent, and building a simple neural network for basic training and testing.  
   _-- Based on the lesson [The spelled-out intro to neural networks and backpropagation: building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0)._ 

2. **[Character-Level Bigram Language Model](https://colab.research.google.com/github/tsilva/aiml-notebooks/blob/main/karpathy-zero-to-hero/002-character-level-bigram-language-model.ipynb)**: This notebook focuses on language modeling, where we manually construct a **bigram language model**, visualize bigram probabilities, and sample from the model. It then transitions into using a neural network to learn bigram probabilities via **backpropagation** and train using **negative log-likelihood loss**, comparing the performance of both manual and learned models.  
   _-- Based on the lesson [The spelled-out intro to language modeling: building makemore](https://www.youtube.com/watch?v=PaCmpygFfXo)._ 

3. **[MLP Character-Level Language Model](https://colab.research.google.com/github/tsilva/aiml-notebooks/blob/main/karpathy-zero-to-hero/003-mlp-character-level-language-model.ipynb)**: This notebook builds a flexible character-level language model using a **Multilayer Perceptron (MLP)**. It covers topics such as embedding character sequences, building and training an MLP, handling numerical stability, **Stochastic Gradient Descent (SGD)**, tuning learning rates, and visualizing learned embeddings.  
   _-- Based on the lesson [Building makemore Part 2: MLP](https://www.youtube.com/watch?v=TCH_1BHY58I)._ 

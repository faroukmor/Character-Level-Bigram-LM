# Character-Level Bigram Language Model from Scratch

A minimalist, character-level Bigram Language Model (LM) built from scratch using **PyTorch**. This project focuses on processing raw text data to train a statistical model that predicts the next character in a sequence and generates new text (names).

## 🚀 Core Features & Concepts Covered

* **Character Tokenizer:** Built custom lookup tables (`stoi` and `itos`) to map characters to integers and vice versa, creating the gateway between text and numerical tensors.
* **Model Smoothing:** Implemented Laplace smoothing by initializing the counts with `(N + 1)` to handle zero-frequency character pairs and prevent infinite loss ($-\infty$) during log evaluation.
* **Probability Normalization:** Utilized PyTorch tensor aggregation and **broadcasting** (`P /= P.sum(1, keepdims=True)`) to cleanly convert raw frequency matrices into structured probability distributions.
* **Evaluation Metric:** Used **Negative Log Likelihood (NLL)** as the loss function to evaluate model performance and measure the quality of the statistical predictions.
* **Text Generation:** Implemented a sampling loop using `torch.multinomial` to feed predictions back into the model, dynamically generating new, unique names character by character.

## 📁 Repository Structure

* `my_first_LM.ipynb` - The complete Jupyter Notebook containing data preprocessing, matrix math, training logic, and evaluation.
* `names.txt` - The dataset file containing thousands of training examples used to build the frequency matrix.

## 🛠️ Requirements

* Python 3.x
* PyTorch

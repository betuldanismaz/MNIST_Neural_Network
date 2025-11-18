# ✍️ MNIST Handwritten Digit Recognition: Deep Learning Hyperparameter Analysis

This repository contains the complete implementation and comprehensive experimental analysis of a fundamental **Multi-Layer Perceptron (MLP)** model applied to the classic **MNIST** (Modified National Institute of Standards and Technology) dataset for **handwritten digit recognition**.

The primary objective of this project was to systematically investigate how changing isolated structural and optimization hyper-parameters impacts the model's final **classification accuracy**.

---

## 🛠️ Project Architecture and Implementation

### Core Implementation
The neural network model is custom-built using a pedagogical approach, leveraging pure **Python** and **NumPy** for core matrix operations. This design choice explicitly focuses on demonstrating the underlying mechanics of the **Stochastic Gradient Descent (SGD)** optimizer and the **Backpropagation** algorithm, rather than relying on high-level frameworks (like TensorFlow or PyTorch).

* **Model Type:** Feed-Forward Neural Network (MLP)
* **Baseline Topology:** 784 Input ➡️ 30 Hidden ➡️ 10 Output
* **Data Structure:** Input images ($28 \times 28$ pixels) are flattened into a **784-dimensional vector**.
* **Code Source:** Adapted from an open-source educational repository by MichalDanielDobrzanski.

---

## 🔬 Experimental Findings and Conclusion

The project compared the Baseline model's performance (Final Accuracy: **94.75%**) against three isolated experiments (Exp. A, B, and C) to identify the most critical factor for performance.

### 1. Experimental Results Summary

| Deney | Parameter Changed | Baseline Topology | Final Accuracy (%) | Baseline Difference |
| :---: | :--- | :---: | :---: | :---: |
| **Baseline** | None | [784, 30, 10] | **94.75** | - |
| **A** | **Mini-Batch Size** ($10 \rightarrow 100$) | [784, 30, 10] | 93.34 | $\mathbf{-1.41}$ |
| **B** | **Depth** (1 $\rightarrow$ 2 Hidden Layers) | [784, 30, 30, 10] | 95.02 | $\mathbf{+0.27}$ |
| **C** | **Capacity** (30 $\rightarrow$ 100 Neurons) | [784, 100, 10] | **96.39** | $\mathbf{+1.64}$ (Best Result) |

### 2. Final Conclusion

The analysis conclusively demonstrated that the most effective parameter change was the increase in **Model Capacity** (Experiment C).

* **Dominance of Capacity:** Widening the network from 30 to 100 hidden neurons yielded the highest performance, achieving **96.39%** accuracy. This confirms that for this shallow MLP architecture, the network's **Representation Power** is the primary limiting factor, and increasing it results in more distinct feature encoding.
* **Cost of Optimization Changes:** Adjusting optimization dynamics (Exp. A: Batch Size 100) negatively impacted accuracy, demonstrating the sensitivity of the Learning Rate ($\eta=3.0$) to changes in the mini-batch size.
* **Marginal Depth Gain:** Incrementally increasing the depth (Exp. B) provided only a minor increase in accuracy, supporting the conclusion that **width** was more beneficial than **depth** for this specific classification task.

---

## ⚙️ Setup and Running the Project

1.  **Environment:** Google Colab (recommended) or any Python 3 environment with `numpy`.
2.  **Files:** All necessary project files (`network.py`, `mnist_loader.py`, `mnist.pkl.gz`) are included in this repository.
3.  **Execution:** Run the Colab notebook sequentially. The code handles data loading, conversion (to prevent iterator exhaustion), and execution of all four training scenarios.

**View the Code:** The primary code for running the experiments is contained within the included Colab notebook (`[Your_Notebook_Name].ipynb`).

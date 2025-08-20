# SWDB 2025 Day 3

Welcome to the **Summer Workshop on the Dynamic Brain (SWDB) 2025** repository focusing on **Dynamical Systems and Reinforcement Learning** approaches for understanding neural behavior and decision-making.

## 🎯 Overview

This repository contains educational materials and computational tools for understanding dynamical systems in neuroscience, with a particular focus on **dynamic foraging task**. This capsule provides hands-on experience with:

- **Basic Reinforcement learning models** for decision-making
- **Model fitting and parameter recovery** for behavioral data
- **Recurrent neural network analysis** for understanding neural dynamics

## 📚 Workshop Structure

### Workshop 1: Model Fitting (`Workshop-1-Model-Fitting.ipynb`)
Learn how to fit computational models to behavioral data, including:
- **Foraging behavior models** (Q-learning, Loss-counting, etc.)
- **Parameter recovery techniques** under the same model architecture
- **Model comparison** of different model architectures using AIC/BIC

### Workshop 2: RNNs for Dynamic Foraging (`Workshop-2-RNNs for Dynamic Foraging.ipynb`)
Explore recurrent neural networks and their dynamics:
- **Actor-critic models** for decision-making
- **Neural trajectory analysis** in hidden state space
- **Dimensionality reduction** (PCA) of neural activity
- **Fixed point analysis** and dynamical systems theory

## Helper Functions

### `utils_model_recovery.py`
Comprehensive toolkit for model fitting and analysis:
- **Foraging agent classes**: `ForagerQLearning`, `ForagerLossCounting`, `BanditModel`
- **Parameter fitting**: Differential evolution optimization
- **Model comparison**: `BanditModelComparison` class
- **Visualization tools**: Parameter recovery plots, confusion matrices

### `utils.py`
Utilities for neural data visualization and analysis:
- **3D trajectory plotting** for neural hidden states
- **Animation tools** for dynamical systems visualization
- **Data processing** helpers

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Jupyter Lab/Notebook
- Basic knowledge of neuroscience and machine learning


## 📁 Repository Structure

```
SWDB_2025_Dynamical_Systems/
├── code/
│   ├── Workshop-1-Model-Fitting.ipynb      # Model fitting tutorial
│   ├── Workshop-2-RNNs for Dynamic Foraging.ipynb  # RNN analysis tutorial
│   ├── utils_model_recovery.py             # Core modeling toolkit
│   ├── utils.py                            # Visualization utilities
│   ├── data/                               # Pre-processed datasets
│   └── resources/                          # Images and diagrams
├── environment/                            # Docker configuration
├── environment.yml                         # Conda environment
└── README.md                              # This file
```

## 📚 Further Links
- [SWDB Day3 GitHub](https://github.com/AllenSWDB/SWDB_2025_Dynamical_Systems)
- [SWDB 2025 wiki](https://github.com/AllenInstitute/swdb_2025_student/wiki)

---

*Happy learning and exploring the fascinating world of RL and dynamical systems in neuroscience! 🧠⚡*
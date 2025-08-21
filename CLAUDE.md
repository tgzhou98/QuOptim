# MCP Gate Optimization Project

## Overview
This is an MCP (Model Context Protocol) server for quantum gate optimization in neutral atom systems. The project provides two main optimization modules:

1. **Pulse-level optimization** (`src/gate_optimize/pulse/`) - Hardware-level gate optimization using GRAPE algorithm
2. **Circuit-level optimization** (`src/gate_optimize/circuit/`) - Machine learning-based circuit optimization using PyTorch

## Project Structure
```
gate_optimize/
├── src/gate_optimize/
│   ├── pulse/           # Hardware pulse optimization
│   │   ├── optimize_cz.py  # CZ gate GRAPE optimization
│   │   └── optimize_x.py   # X gate GRAPE optimization
│   ├── circuit/         # ML circuit optimization
│   │   ├── dqn.py       # Deep Q-Network implementation
│   │   ├── ppo.py       # Proximal Policy Optimization
│   │   ├── environment.py # RL environment for circuit optimization
│   │   └── ...          # Other circuit optimization modules
│   └── server.py        # MCP server implementation
├── model/               # Trained models and evaluation data
│   ├── plots/          # Training plots and results
│   ├── eval/           # Evaluation parameters and test cases
│   └── tests/          # Test configurations
└── pyproject.toml      # Project dependencies
```

## MCP Tools Available
- `optimize_cz_gate`: Optimizes CZ gate using GRAPE algorithm
- `optimize_x_gate`: Optimizes X gate using GRAPE with Fourier coefficients

## Dependencies
- PyTorch (for neural network models)
- NumPy
- Matplotlib
- MCP FastMCP framework

## Usage
Run the MCP server to access quantum gate optimization tools for neutral atom quantum computing systems.
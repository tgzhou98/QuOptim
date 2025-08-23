# MCP Gate Optimize

![CI](https://github.com/tgzhou98/QuOptim/workflows/CI/badge.svg)

A Model Context Protocol (MCP) server for quantum gate optimization in neutral atom systems.

## Features

### Pulse-Level Optimization (`src/gate_optimize/pulse/`)
- **CZ Gate Optimization**: GRAPE algorithm implementation for controlled-Z gate optimization
- **X Gate Optimization**: Robust GRAPE with Fourier coefficients for X gate optimization

### Circuit-Level Optimization (`src/gate_optimize/circuit/`)
- **Deep Q-Network (DQN)**: Reinforcement learning for circuit optimization
- **Proximal Policy Optimization (PPO)**: Advanced RL algorithm for gate sequence optimization
- **Environment Simulation**: Custom RL environment for quantum circuit optimization

### qec for any quantum code (`src/gate_optimize/qec/`)
- **Quantum error correction**: run decoder for the quantum code

## Project Structure
```
gate_optimize/
├── src/gate_optimize/
│   ├── pulse/           # Hardware pulse optimization
│   ├── circuit/         # ML circuit optimization  
│   ├── qec/             # error correction code 
│   └── server.py        # MCP server implementation
├── model/               # Trained models and evaluation data
│   ├── plots/          # Training results and visualizations
│   ├── eval/           # Evaluation parameters and benchmarks
│   └── tests/          # Test configurations and datasets
└── pyproject.toml      # Project dependencies
```

## MCP Tools
- `optimize_cz_gate`: Optimize CZ gate using GRAPE algorithm
- `optimize_x_gate`: Optimize X gate using GRAPE with Fourier coefficients

## Installation
```bash
uv sync
```

## Usage
Start the MCP server to access quantum gate optimization tools for neutral atom quantum computing research.

## Command

### Circuit generation

Some simple test

- Generate circuit from the GHZ/Steane code `model/eval/7bit_params.json`
`python -m gate_optimize.circuit.minimal_runner`

- Optimize the CZ gate
`python -m gate_optimize.pulse.optimize_cz`

- Optimize the X gate
`python -m gate_optimize.pulse.optimize_x`



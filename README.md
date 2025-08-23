# MCP Gate Optimize

![CI](https://github.com/tgzhou98/QuOptim/workflows/CI/badge.svg)

An MCP (Model Context Protocol) server with a built‑in GUI for pulse- and circuit‑level quantum optimization targeting neutral‑atom systems.

- **Pulse level**: GRAPE‑based CZ and robust X gate design with visualization and robustness analysis.
- **Circuit level**: RL‑assisted circuit synthesis plus Qiskit baselines, timeline plotting, transpile‑based simplification, and fidelity comparison under either a physical noise model or calibrated data.
- **Monitoring GUI**: A PyQt6 dashboard receives live updates from tools (plots, fidelity traces, text summaries) via a local Flask endpoint.

## Features

### Pulse‑Level Optimization (`src/gate_optimize/pulse/`)
- **CZ gate (GRAPE)**: Bidirectional evolution, live pulse shape/fidelity plots.
- **X gate (Robust GRAPE, Fourier)**: Envelope shaping, 3×3 optimization grid, 11×11 robustness map, Bloch‑sphere trajectories.

### Circuit‑Level Optimization (`src/gate_optimize/circuit/`)
- **RL policies**: DQN/PPO/VPG support through `Experiment` helpers and pre‑trained models.
- **Baselines**: Multiple Qiskit synthesizers (AG, BM, greedy, Bravyi) for comparison.
- **Timeline plotting**: Per‑qubit execution visualization for bottleneck analysis.
- **Transpile simplification**: Depth reduction over a constrained basis.
- **Fidelity simulation**: Physical model or calibrated model based on analyzed benchmark data.

### QEC Utilities (`src/gate_optimize/qec/`)
- Tools for code construction, decoding, and visualization (see module docs and tests).

## Project Structure
```
gate_optimize/
├── src/gate_optimize/
│   ├── pulse/            # Hardware pulse optimization
│   ├── circuit/          # ML circuit optimization
│   ├── qec/              # Error correction utilities
│   ├── server.py         # MCP tools implementation
│   ├── __init__.py       # GUI + MCP stdio launcher (main)
│   └── __main__.py       # python -m gate_optimize entry
├── model/
│   ├── plots/            # Trained RL policies and training artifacts
│   └── eval/             # Benchmarks/params (e.g., 7‑bit GHZ/Steane)
└── pyproject.toml        # Dependencies and console script
```

## Available MCP Tools
All tools stream progress and images to the GUI and return results to the MCP client.

- `optimize_cz_gate` (pulse): GRAPE CZ gate optimization with live pulse/fidelity.
- `optimize_x_gate` (pulse): Robust Fourier‑parameterized X gate with robustness analysis and Bloch‑sphere trajectories.
- `generate_circuits` (circuit): Create RL and Qiskit baseline circuits from stabilizer generators.
- `plot_timeline` (circuit): Visualize per‑qubit execution timelines of a selected circuit.
- `simplify_best_circuit` (circuit): Pick the fewest‑gates circuit, transpile, and compare before/after.
- `compare_circuits_fidelity` (circuit): Evaluate circuits under a physical noise model or calibrated error rates.
- `simulate_gate_benchmark_data` (calibration): Generate standardized RB‑style synthetic data for X and CZ.
- `analyze_gate_fidelity_from_data` (calibration): Fit benchmark data to extract 1Q/2Q fidelities and SPAM error.

## End‑to‑end workflow: from code to experiment

1) Discover code and load stabilizers (external service)
   - Use a separate code‑discovery server (placeholder: *******; WIP) to extract canonical stabilizer generators for the selected code.
   - Output: validated stabilizer list for the target code.

2) Reference logical vs physical error rates
   - Provide a reference curve/table of logical error rate versus physical error rate for the code to set expectations before compilation (leveraging QEC tooling).

3) Generate candidate preparation circuits
   - Run `generate_circuits` to produce RL‑optimized and Qiskit baseline circuits (QASM3 and gate counts). Optionally use `simplify_best_circuit` to reduce depth.


4) Evaluate circuits with a calibrated noise model
   - Use `compare_circuits_fidelity` to estimate each circuit’s fidelity under a realistic physical model and shortlist top candidates.

<div align="center">
<table width="720" border="1" cellspacing="0" cellpadding="8">
<tr><td>
<strong>Calibration and feedback iteration loop</strong>
<ul>
<li>Ingest experimental benchmark data; run <code>analyze_gate_fidelity_from_data</code> to extract calibrated 1Q/2Q fidelities and SPAM.</li>
<li>Feed calibrated rates into <code>compare_circuits_fidelity</code> for realistic circuit ranking.</li>
<li>Revise QEC logical‑vs‑physical assumptions as needed.</li>
<li>Use calibrated metrics as objectives for later pulse optimization (<code>optimize_x_gate</code>, <code>optimize_cz_gate</code>).</li>
</ul>
</td></tr>
</table>
</div>





5) Prepare experimental execution references
   - Use `plot_timeline` to visualize per‑qubit scheduling, parallelism, and bottlenecks.

6) Hardware pulse optimization
   - Optimize X and CZ gate pulses (`optimize_x_gate`, `optimize_cz_gate`) and review pulse shapes, convergence, robustness, and Bloch‑sphere trajectories.

## Installation
Requires Python ≥ 3.13.

```bash
uv sync
```

## Running the MCP Server + GUI
The project exposes a console script and a module entrypoint; both start the MCP stdio server and the monitoring GUI:

```bash
# Using the console script
uv run mcp-gate-optimize

# Or as a module
uv run python -m gate_optimize
```

The GUI listens for updates on `http://127.0.0.1:12345/update` and displays:
- live fidelity traces
- primary result images
- text summaries from tools

You can connect any MCP‑capable client (e.g., CLI or IDE integrations) to the stdio server to invoke tools.

## Typical Workflows

### Pulse optimization
- Run `optimize_cz_gate` or `optimize_x_gate` from your MCP client to optimize pulses; watch the GUI for live plots.

### Circuit synthesis and comparison
1) `generate_circuits` with your stabilizer generators (e.g., GHZ/Steane) to obtain RL + Qiskit variants (QASM3).
2) Optionally `plot_timeline` for a chosen index to analyze scheduling and bottlenecks.
3) `simplify_best_circuit` to minimize gate count using transpile.
4) `compare_circuits_fidelity` to rank circuits by simulated fidelity (physical model by default).

### Calibration‑aware evaluation
1) `simulate_gate_benchmark_data` to create standardized RB‑style data for X and CZ.
2) `analyze_gate_fidelity_from_data` to fit and extract calibrated 1Q/2Q fidelities and SPAM error.
3) Pass the returned JSON to `compare_circuits_fidelity` for calibrated circuit ranking.

## Notes on RL models
`generate_circuits` expects a pre‑trained RL model at a path under `model/plots/<...>/model.pkl` that matches the parameters in `model/eval/7bit_params.json`. If not found, the tool reports a clear error. You can adjust parameters, provide your own trained model, or rely solely on Qiskit baselines.

## Standalone module commands
For quick experiments without MCP:

- Generate GHZ/Steane circuits from `model/eval/7bit_params.json`:
  ```bash
  uv run python -m gate_optimize.circuit.minimal_runner
  ```
- Optimize the CZ gate:
  ```bash
  uv run python -m gate_optimize.pulse.optimize_cz
  ```
- Optimize the X gate:
  ```bash
  uv run python -m gate_optimize.pulse.optimize_x
  ```

## Testing
```bash
uv run pytest -q
```


## Julia MCP Server
Julia MCP server can generate code stabilizers and logical operators, and compute code distance.

### Installation
First, install&nbsp;<a href="https://julialang.org"><img src="https://raw.githubusercontent.com/JuliaLang/julia-logo-graphics/master/images/julia.ico" width="16em"> Julia </a> &nbsp;Programming Language. Then run
```bash
julia --project=juliamcp -e "using Pkg; Pkg.instantiate()"
```
to install the dependencies. Finally, add the following to your MCP client configuration:
```json
"command": "julia",
"args": ["--project=/path/to/project/juliamcp", "/path/to/project/juliamcp/server.jl"]
```

#!/usr/bin/env python3
"""
Minimal runner for printing RL quantum circuit diagrams to stdout.
Focuses only on circuit generation for GHZ/7-qubit Steane code with fidelity calculations.
Also provide the Qiskit benchmarks for comparison.

Usage:
    python -m gate_optimize.circuit.minimal_runner
    OR
    From project root: python src/gate_optimize/circuit/minimal_runner.py
"""

import os
import sys
from io import StringIO
import json
import numpy as np
import time
from pathlib import Path
from importlib import resources

# Handle both direct execution and module execution
if __name__ == '__main__' and __package__ is None:
    # Direct execution - add parent directories to path
    try:
        project_root = resources.files('gate_optimize').parent.parent
    except (ImportError, AttributeError):
        # Fallback for older Python or if resources fails
        project_root = Path(__file__).resolve().parent.parent.parent.parent
    sys.path.insert(0, str(project_root / 'src'))
    from gate_optimize.circuit.environment import Environment
    from gate_optimize.circuit.experiment import Experiment
    from gate_optimize.circuit import utils
    from gate_optimize.circuit.runner import parse
else:
    # Module execution
    from .environment import Environment
    from .experiment import Experiment
    from . import utils
    from .runner import parse

from qiskit import QuantumCircuit
import qiskit.quantum_info as qi
import qiskit.synthesis as qs
import torch
import stim
import tqdm

def bravyi_circuit(stabilizer_generators: list[str]) -> QuantumCircuit:
    """Create circuit using Qiskit's Bravyi synthesis."""
    gens = [l.replace('_', 'I') for l in stabilizer_generators]
    return qi.StabilizerState.from_stabilizer_list(gens).clifford.to_circuit()

def get_qiskit_benchmarks(stabilizer_generators: list[str]):
    """Get Qiskit benchmark circuits for comparison."""
    # Convert stabilizers to Clifford
    gens = [l.replace('_', 'I') for l in stabilizer_generators]
    cliff = qi.StabilizerState.from_stabilizer_list(gens).clifford
    
    # Get different synthesis methods
    bravyi_qc = bravyi_circuit(stabilizer_generators)
    ag_qc = qs.synth_clifford_ag(cliff)
    bm_qc = qs.synth_clifford_full(cliff)
    greedy_qc = qs.synth_clifford_greedy(cliff)
    
    return {
        'bravyi': bravyi_qc,
        'ag': ag_qc,
        'bm': bm_qc,
        'greedy': greedy_qc
    }

def print_rl_circuit(actions, env, fidelity, target_tableau):
    """Print fidelity, circuit diagram, and Qiskit benchmarks."""
    if len(actions) == 0:
        return
    
    # Create the inverted circuit using environment method
    try:
        qc = env.get_inverted_ckt(actions)
    except (IndexError, AttributeError):
        # Fallback: build circuit manually
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(env.qubits)
        
        for action in actions:
            action_idx = action.item() if hasattr(action, 'item') else action
            if action_idx < len(env.gates):
                gate_name = env.gates[action_idx]
                # Simple manual gate application for common gates
                if gate_name.startswith('h('):
                    qubit = int(gate_name.split('(')[1].split(')')[0])
                    qc.h(qubit)
                elif gate_name.startswith('cnot('):
                    qubits = gate_name.split('(')[1].split(')')[0].split(',')
                    qc.cx(int(qubits[0]), int(qubits[1]))
                elif gate_name.startswith('s('):
                    qubit = int(gate_name.split('(')[1].split(')')[0])
                    qc.s(qubit)
    
    # Print RL results
    print(f"RL Fidelity: {fidelity:.6f}")
    print("RL Circuit Diagram:")
    print(qc.draw('text', fold=-1))
    
    gate_names = [env.gates[a.item() if hasattr(a, 'item') else a] for a in actions]
    print("RL Gate Array:", gate_names)
    print(f"RL Gate Count: {len(qc.data)}")
    
    # Get and print Qiskit benchmarks
    stabilizer_strings = [str(s) for s in target_tableau.to_stabilizers()]
    benchmarks = get_qiskit_benchmarks(stabilizer_strings)
    
    print("\n--- QISKIT BENCHMARKS ---")
    
    for method_name, benchmark_qc in benchmarks.items():
        print(f"\n{method_name.upper()} Circuit:")
        print(f"Gate Count: {len(benchmark_qc.data)}")
        print("Circuit Diagram:")
        print(benchmark_qc.draw('text', fold=-1))
        
        # Get gate array for benchmark using modern Qiskit API
        gate_array = []
        for instr in benchmark_qc.data:
            # Use modern Qiskit API
            gate_name = instr.operation.name
            qubits = instr.qubits
            
            try:
                # Extract qubit indices by finding their position in the circuit
                qubit_indices = []
                for qubit in qubits:
                    # Find the index of this qubit in the circuit's qubit list
                    for i, circuit_qubit in enumerate(benchmark_qc.qubits):
                        if circuit_qubit == qubit:
                            qubit_indices.append(i)
                            break
                    else:
                        # If not found, use 0 as fallback
                        qubit_indices.append(0)
                
                # Format the gate
                if len(qubit_indices) == 1:
                    gate_array.append(f"{gate_name}({qubit_indices[0]})")
                elif len(qubit_indices) == 2:
                    gate_array.append(f"{gate_name}({qubit_indices[0]},{qubit_indices[1]})")
                else:
                    gate_array.append(f"{gate_name}({','.join(map(str, qubit_indices))})")
                    
            except Exception as e:
                # If all else fails, just show the gate name
                gate_array.append(f"{gate_name}")
                
        print(f"Gate Array: {gate_array}")
    
    print("\n" + "="*80)
    return qc

def load_steane_targets(testfile):
    """Load Steane code target states from file."""
    # Convert relative path to absolute path from project root
    if testfile.startswith('../../../'):
        try:
            project_root = resources.files('gate_optimize').parent.parent
        except (ImportError, AttributeError):
            # Fallback for older Python or if resources fails
            project_root = Path(__file__).resolve().parent.parent.parent.parent
        testfile_path = os.path.join(project_root, testfile.replace('../../../', ''))
    else:
        testfile_path = testfile
    
    if not os.path.exists(testfile_path):
        print(f"Test file not found: {testfile_path}")
        return []
    
    with open(testfile_path, 'r') as f:
        content = f.read().strip()
    
    # Parse the stabilizer generators - each line is a list of stabilizer strings
    lines = content.split('\n')
    target_tableaus = []
    for line in lines:
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            # Parse the stabilizer strings and convert to tableau
            stabilizer_strings = eval(line)
            tableau = stim.Tableau.from_stabilizers([stim.PauliString(s) for s in stabilizer_strings])
            target_tableaus.append(tableau)
    
    return target_tableaus

def run_minimal_circuit_generation():
    """Run circuit generation with clean output - only fidelity and diagrams."""
    # Suppress debug output by redirecting stderr temporarily
    
    # Parse parameters silently  
    # Use resources to get project root relative to gate_optimize package
    try:
        project_root = resources.files('gate_optimize').parent.parent
    except (ImportError, AttributeError):
        # Fallback for older Python or if resources fails
        project_root = Path(__file__).resolve().parent.parent.parent.parent
    params_path = str(project_root / 'model' / 'eval' / '7bit_params.json')
    
    # Capture the noisy output during parsing
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()
    
    all_results = []
    
    try:
        args = parse(['-fromjson', params_path])
        
        # Set up minimal globals
        utils._globals = {
            'debug': False,  # Force debug off
            'dist': args.dist,
            'rewardtype': args.rewardtype,
            'swanlab': False,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'noise': lambda state: state,
            'bufsize': args.bufsize,
            'gamma': args.gamma,
            'tau': args.tau,
            'num_envs': args.num_envs
        }
        utils.args = args
        
        # Load target states
        target_tableaus = load_steane_targets(args.testfile)
        if not target_tableaus:
            return
        
        # Set up experiment
        utils.set_seed(args.seed)
        target_state = stim.Tableau(args.qbits)
        exp = Experiment(args.a, training_req=False, n_workers=1)
        
        # Load trained model
        try:
            project_root = resources.files('gate_optimize').parent.parent
        except (ImportError, AttributeError):
            # Fallback for older Python or if resources fails
            project_root = Path(__file__).resolve().parent.parent.parent.parent
        model_dir = project_root / 'model' / 'plots' / f"{args.qbits}-{args.tol}-{args.name}--{args.exptdate}"
        model_path = str(model_dir / 'model')
        if not os.path.exists(model_path + '.pkl'):
            return
        
        # Process each target state
        for i, target_tableau in enumerate(target_tableaus):
            # Initialize test environment first to create sample_env
            env = exp.initialize_test_env(target_state, target_tableau, args.tol, args.maxsteps, args.gateset, args.dist)
            env.max_steps = int(1 * args.maxsteps)
            
            # Initialize agent if not already done
            if not hasattr(exp, 'agent'):
                if args.a in ['ppo', 'vpg']:
                    exp.initialize_agent_pg(
                        policy_hidden=args.phidden,
                        policy_activ_fn=getattr(torch.nn.functional, args.activfn),
                        policy_model_max_grad_norm=0.5,
                        policy_optimizer_fn=lambda net: torch.optim.Adam(net.parameters(), lr=args.plr),
                        value_hidden=args.vhidden,
                        value_activ_fn=getattr(torch.nn.functional, args.activfn),
                        value_model_max_grad_norm=0.5,
                        value_optimizer_fn=lambda net: torch.optim.Adam(net.parameters(), lr=args.vlr),
                        entropy_loss_weight=args.entropywt,
                        gamma=args.gamma,
                    )
                
                # Load the trained model
                exp.load_model(model_path)
            
            # Generate circuits using the trained RL agent
            tries = 5
            best_circuits, _ = exp.evaluate(env, n_eps=tries, num_best=tries, verbose=0)
            all_results.append((target_tableau, env, best_circuits))
            
    finally:
        # Restore stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    # Now print only the clean results
    for i, (target_tableau, env, best_circuits) in enumerate(all_results):
        print(f"TARGET STATE {i+1}")
        print(f"Stabilizers: {[str(s) for s in target_tableau.to_stabilizers()]}")
        
        for j, (actions, _, _, fidelity) in enumerate(best_circuits):
            print(f"\nCircuit {j+1}:")
            print_rl_circuit(actions, env, fidelity, target_tableau)

def main():
    """Main entry point."""
    try:
        run_minimal_circuit_generation()
    except KeyboardInterrupt:
        print("Interrupted")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
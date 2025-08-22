import base64
from typing import Annotated, Literal, List
import requests
import json

from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent

from .pulse.optimize_cz import GRAPE as GRAPE_CZ
from .pulse.optimize_x import RobustGRAPEFourierCoeffs as GRAPE_X
from .circuit.minimal_runner import run_minimal_circuit_generation, load_steane_targets
from .circuit.environment import Environment
from .circuit.experiment import Experiment
from .circuit import utils
import numpy as np
from matplotlib import pyplot as plt

import stim
import tempfile
import os
import torch
import io
import base64


mcp = FastMCP("mcp-gate-optimize")

GUI_ENDPOINT = "http://127.0.0.1:12345/update"

def send_to_gui(**kwargs):
    """一个辅助函数，用于将结构化数据发送到自定义GUI"""
    try:
        kwargs.setdefault("status", "finished")
        requests.post(GUI_ENDPOINT, json=kwargs, timeout=2)
    except requests.exceptions.RequestException:
        pass


@mcp.tool(
    name="optimize_cz_gate",
    description="Run the GRAPE algorithm to optimize the CZ gate.",
)
async def optimize_cz_gate(
    iterations: Annotated[int, "The number of iterations for the optimization."] = 500,
    learning_rate: Annotated[float, "The learning rate for the optimization."] = 0.5,
) -> list[ImageContent | TextContent]:
    """
    Run the GRAPE algorithm to optimize the CZ gate.
    """
    g = GRAPE_CZ()
    fidelity_history = []
    update_interval = 10
    live_img_base64 = ""
    fidelity = 0.0

    for i in range(iterations):
        fidelity = g.iteration_onestep(learning_rate)
        
        is_update_step = (i % update_interval == 0)
        if is_update_step:
            fidelity_history.append(fidelity)

        if is_update_step or i == iterations - 1:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(10, 6))
            time_steps_stair0, pwc_pulse_stair0 = g.PWC_pulse(g.phi)
            ax.plot(time_steps_stair0, pwc_pulse_stair0, "b-", linewidth=2)
            ax.set_xlabel("time")
            ax.set_ylabel("pulse strength")
            ax.set_title(f"CZ Gate Pulse Shape (Iteration {i + 1})")
            ax.set_ylim([0, 2 * np.pi])
            ax.grid(True, alpha=0.3)
            
            import io
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            live_img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
            img_buffer.close()
            plt.close(fig)

            send_to_gui(
                status="running" if i < iterations - 1 else "finished",
                tool_name="optimize_cz_gate",
                parameters={"iterations": iterations, "learning_rate": learning_rate},
                live_metrics={"iteration": i + 1, "fidelity": fidelity},
                fidelity_history=fidelity_history,
                main_images=[live_img_base64]
            )
            
    final_text_for_lucien = f"Final fidelity: {fidelity:.6f}"

    return [
        ImageContent(type="image", data=live_img_base64, mimeType="image/png"),
        TextContent(type="text", text=final_text_for_lucien),
    ]


@mcp.tool(
    name="optimize_x_gate",
    description="Run the GRAPE algorithm to optimize the X gate with Fourier coefficients.",
)
async def optimize_x_gate(
    iterations: Annotated[int, "The number of iterations for the optimization."] = 100,
    learning_rate: Annotated[float, "The learning rate for the optimization."] = 0.05,
    fourier_terms: Annotated[int, "The number of Fourier terms to optimize."] = 6,
) -> list[ImageContent | TextContent]:
    """
    Run the GRAPE algorithm to optimize the X gate with Fourier coefficients.
    """
    g = GRAPE_X(num_fourier_terms=fourier_terms)
    fidelity_history = []
    update_interval = 5
    live_img_base64 = ""
    avg_fidelity_opt_grid = 0.0

    for i in range(iterations):
        avg_fidelity_opt_grid, _ = g.iteration_onestep_numerical_derivative(lr=learning_rate)
        
        is_update_step = (i % update_interval == 0)
        if is_update_step:
            fidelity_history.append(avg_fidelity_opt_grid)

        if is_update_step or i == iterations - 1:
            plt.close('all')
            fig, ax = plt.subplots(figsize=(10, 6))
            t_plot_live = np.linspace(0, g.t_final, 400)
            phi_plot_live = g.reconstruct_phi_at_t(t_plot_live, g.a_coeffs, g.b_coeffs)
            ax.plot(t_plot_live, phi_plot_live, "r-", linewidth=2, label="Phase φ(t) (Fourier)")
            ax.set_xlabel("time")
            ax.set_ylabel("Phase φ(t) (radians)", color="r")
            ax.tick_params(axis="y", labelcolor="r")
            ax.set_title(f"X Gate Pulse Shape (Iteration {i + 1})")
            ax.set_ylim([0, 2 * np.pi])
            ax.grid(True, alpha=0.3)
            ax.legend()

            import io
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            live_img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
            img_buffer.close()
            plt.close(fig)

            send_to_gui(
                status="running" if i < iterations - 1 else "finished",
                tool_name="optimize_x_gate",
                parameters={"iterations": iterations, "learning_rate": learning_rate, "fourier_terms": fourier_terms},
                live_metrics={"iteration": i + 1, "fidelity": avg_fidelity_opt_grid},
                fidelity_history=fidelity_history,
                main_images=[live_img_base64]
            )
    
    final_text_for_lucien = f"Final average fidelity: {g.fidelity:.6f}"

    return [
        ImageContent(type="image", data=live_img_base64, mimeType="image/png"),
        TextContent(type="text", text=final_text_for_lucien),
    ]


@mcp.tool(
    name="generate_circuit_from_stabilizers",
    description="Generate optimized quantum circuits from stabilizer codes using RL and compare with Qiskit benchmarks.",
)
async def generate_circuit_from_stabilizers(
    stabilizers: Annotated[List[str], "List of stabilizer strings for the quantum error correction code (e.g., ['+ZZ_____', '+_ZZ____', '+XXXXXXX'] for 7-qubit Steane code)"],
    num_circuits: Annotated[int, "Number of different circuit optimizations to generate"] = 5,
) -> list[ImageContent | TextContent]:
    """
    Generate optimized quantum circuits from stabilizer generator strings.
    """
    if not stabilizers or not isinstance(stabilizers, list):
        error_text = "Error: stabilizers must be a non-empty list of strings"
        send_to_gui(status="error", text_result=error_text)
        return [TextContent(type="text", text=error_text)]
    
    clean_stabilizers = [s.lstrip('+-').replace('_', 'I') for s in stabilizers]
    num_qubits = len(clean_stabilizers[0])
    target_tableau = stim.Tableau.from_stabilizers(
        [stim.PauliString(s) for s in clean_stabilizers], 
        allow_underconstrained=True
    )

    from importlib import resources
    try:
        project_root = resources.files('gate_optimize').parent.parent
    except (ImportError, AttributeError):
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent.parent.parent
    
    params_path = str(project_root / 'model' / 'eval' / '7bit_params.json')
    from .circuit.runner import parse
    args = parse(['-fromjson', params_path])
    
    args.qbits = num_qubits
    utils.set_seed(args.seed)
    
    utils._globals = {
        'debug': False, 'dist': args.dist, 'rewardtype': args.rewardtype,
        'swanlab': False, 'device': utils.get_device(prefer_cuda=True),
        'noise': lambda state: state, 'bufsize': args.bufsize,
        'gamma': args.gamma, 'tau': args.tau, 'num_envs': 1
    }
    utils.args = args
    
    exp = Experiment(args.a, training_req=False, n_workers=1)
    
    results = []
    images = []
    results.append("QUANTUM CIRCUIT GENERATION FROM STABILIZERS")
    results.append("=" * 60)
    results.append(f"Input Stabilizers: {stabilizers}")
    results.append(f"Number of Qubits: {num_qubits}")
    results.append(f"Algorithm: {args.a}")
    results.append(f"Gateset: {args.gateset}")
    results.append(f"Max Steps: {args.maxsteps}")
    results.append("")
    
    model_dir = project_root / 'model' / 'plots' / f"{args.qbits}-{args.tol}-{args.name}--{args.exptdate}"
    model_path = str(model_dir / 'model')
    
    model_exists = os.path.exists(model_path + '.pkl')
    if model_exists:
        results.append(f"✓ Found pre-trained model at {model_path}.pkl")
    else:
        results.append(f"⚠ No pre-trained model found at {model_path}.pkl")

    target_state = stim.Tableau(args.qbits)
    env = exp.initialize_test_env(target_state, target_tableau, args.tol, args.maxsteps, args.gateset, args.dist)
    env.max_steps = int(1 * args.maxsteps)
    
    if model_exists and not hasattr(exp, 'agent'):
        results.append("🔄 Initializing RL agent...")
        if args.a in ['ppo', 'vpg']:
            exp.initialize_agent_pg(
                policy_hidden=args.phidden, policy_activ_fn=getattr(torch.nn.functional, args.activfn),
                policy_model_max_grad_norm=0.5, policy_optimizer_fn=lambda net: torch.optim.Adam(net.parameters(), lr=args.plr),
                value_hidden=args.vhidden, value_activ_fn=getattr(torch.nn.functional, args.activfn),
                value_model_max_grad_norm=0.5, value_optimizer_fn=lambda net: torch.optim.Adam(net.parameters(), lr=args.vlr),
                entropy_loss_weight=args.entropywt, gamma=args.gamma,
            )
        results.append("🔄 Loading trained model...")
        exp.load_model(model_path)
        results.append("✅ Successfully loaded pre-trained RL model!")

    results.append(f"\n🔄 Generating {min(num_circuits, 5)} circuit variants...")
    best_circuits, _ = exp.evaluate(env, n_eps=min(num_circuits, 5), num_best=min(num_circuits, 5), verbose=0)
    
    for circuit_num, (actions, _, _, final_fidelity) in enumerate(best_circuits):
        results.append(f"\n--- CIRCUIT VARIANT {circuit_num + 1} ---")
        
        try:
            int_actions = [a.item() if hasattr(a, 'item') else int(a) for a in actions]
            
            try:
                qc = env.get_inverted_ckt(int_actions)
            except (IndexError, AttributeError):
                from qiskit import QuantumCircuit
                qc = QuantumCircuit(env.qubits)
                for action in reversed(int_actions):
                    if action < len(env.gates):
                        gate_name = env.gates[action]
                        if gate_name.startswith('h('):
                            qubit = int(gate_name.split('(')[1].split(')')[0])
                            qc.h(qubit)
                        elif gate_name.startswith('cnot('):
                            qubits = gate_name.split('(')[1].split(')')[0].split(',')
                            qc.cx(int(qubits[0]), int(qubits[1]))

            gate_names = [env.gates[a] for a in int_actions if a < len(env.gates)]
            results.append(f"RL Fidelity: {final_fidelity:.6f}")
            results.append("RL Circuit Diagram:")
            results.append(str(qc.draw('text', fold=80)))
            results.append(f"RL Gate Array: {gate_names}")
            results.append(f"RL Gate Count: {len(qc.data)}")
            
            # Generate circuit diagram
            plt.close('all')
            fig, ax = plt.subplots(figsize=(12, 6))
            qc.draw('mpl', ax=ax)
            ax.set_title(f"RL Circuit Variant {circuit_num + 1} - Fidelity: {final_fidelity:.6f}, Gate Count = {len(qc.data)}")
            
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
            img_buffer.close()
            plt.close(fig)
            images.append(ImageContent(type="image", data=img_base64, mimeType="image/png"))
            
            # Generate timeline plot
            try:
                from .circuit.utils import plot_timeline
                timeline_img_base64 = plot_timeline(int_actions, env)
                images.append(ImageContent(type="image", data=timeline_img_base64, mimeType="image/png"))
                results.append("✅ Generated execution timeline visualization")
            except Exception as timeline_error:
                results.append(f"⚠ Timeline generation error: {timeline_error}")

        except Exception as e:
            results.append(f"Error during circuit processing for variant {circuit_num + 1}: {e}")
            
    import qiskit.quantum_info as qi
    import qiskit.synthesis as qs
    results.append(f"\n--- QISKIT BENCHMARKS ---")
    stabilizer_strings = [s.lstrip('+-').replace('_', 'I') for s in stabilizers]
    max_length = max(len(s) for s in stabilizer_strings)
    actual_num_qubits = max(num_qubits, max_length)
    padded_stabilizers = [s + 'I' * (actual_num_qubits - len(s)) for s in stabilizer_strings]
    cliff = qi.StabilizerState.from_stabilizer_list(padded_stabilizers, allow_underconstrained=True).clifford
    benchmarks = {
        'bravyi': cliff.to_circuit(), 'ag': qs.synth_clifford_ag(cliff),
        'bm': qs.synth_clifford_full(cliff), 'greedy': qs.synth_clifford_greedy(cliff)
    }
    for method_name, benchmark_qc in benchmarks.items():
        results.append(f"\n{method_name.upper()} Circuit:")
        results.append(f"Gate Count: {len(benchmark_qc.data)}")
        if len(benchmark_qc.data) < 30:
            results.append("Circuit Diagram:")
            results.append(str(benchmark_qc.draw('text', fold=80)))
            plt.close('all')
            fig, ax = plt.subplots(figsize=(12, 6))
            benchmark_qc.draw('mpl', ax=ax)
            ax.set_title(f"{method_name.upper()} Circuit - Gate Count: {len(benchmark_qc.data)}")
            import io
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
            img_buffer.close()
            plt.close(fig)
            images.append(ImageContent(type="image", data=img_base64, mimeType="image/png"))
    
    results.append(f"\n{'=' * 60}")

    final_text = "\n".join(results)
    images_base64 = [img.data for img in images]
    
    send_to_gui(
        status="finished",
        tool_name="generate_circuit_from_stabilizers",
        parameters={"stabilizers": stabilizers, "num_circuits": num_circuits},
        text_result=final_text,
        main_images=images_base64
    )
    
    content_list = [TextContent(type="text", text=final_text)]
    if images:
        content_list.extend(images)
        
    return content_list
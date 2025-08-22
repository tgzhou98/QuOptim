import base64
from typing import Annotated, Literal, List
import requests # --- 新增代码 ---
import json     # --- 新增代码 ---

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


mcp = FastMCP("mcp-gate-optimize")

# --- 新增代码开始 ---
# 定义GUI的API端点
GUI_ENDPOINT = "http://127.0.0.1:12345/update"

def send_to_gui(text_content: str, images_base64: List[str]):
    """一个辅助函数，用于将数据发送到自定义GUI"""
    try:
        payload = {
            "text": text_content,
            "images": images_base64
        }
        # 使用较低的超时时间，以防GUI未运行导致长时间阻塞
        requests.post(GUI_ENDPOINT, json=payload, timeout=2)
    except requests.exceptions.RequestException:
        # 如果GUI没有运行，此函数将静默失败，不会影响主程序流程
        pass
# --- 新增代码结束 ---


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

    Args:
        iterations: The number of iterations for the optimization.
        learning_rate: The learning rate for the optimization.

    Returns:
        A list containing the plot of the optimized pulses and the final fidelity.
    """
    g = GRAPE_CZ()
    infid = []
    
    # --- 新增代码: 定义更新频率 ---
    update_interval = 10  # 每 10 次迭代更新一次GUI

    for i in range(iterations):
        fidelity = g.iteration_onestep(learning_rate)
        infid.append(abs(1 - fidelity))

        # --- 新增代码块: 周期性发送实时更新到GUI ---
        # 在每个更新间隔的第一次迭代或最后一次迭代时触发
        if i % update_interval == 0 or i == iterations - 1:
            # 1. 准备实时文本
            live_text = (f"Iteration: {i + 1}/{iterations}\n"
                         f"Current Fidelity: {fidelity:.6f}")

            # 2. 生成实时脉冲图像
            plt.close('all')
            fig, ax = plt.subplots(figsize=(10, 6))
            time_steps_stair0, pwc_pulse_stair0 = g.PWC_pulse(g.phi)
            ax.plot(time_steps_stair0, pwc_pulse_stair0, "b-", linewidth=2)
            ax.set_xlabel("time")
            ax.set_ylabel("pulse strength")
            ax.set_title(f"Optimization in Progress... (Iteration {i + 1})")
            ax.set_ylim([0, 2 * np.pi])
            ax.grid(True, alpha=0.3)
            
            # 3. 将图像转换为base64
            import io
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            live_img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
            img_buffer.close()
            plt.close(fig)

            # 4. 发送到GUI
            send_to_gui(text_content=live_text, images_base64=[live_img_base64])
        # --- 新增代码块结束 ---

    # --- 以下是原始代码的结尾部分，保持不变 ---
    # 它负责生成最终的图像并返回给Lucien
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 6))
    time_steps_stair0, pwc_pulse_stair0 = g.PWC_pulse(g.phi)
    ax.plot(time_steps_stair0, pwc_pulse_stair0, "b-", linewidth=2)
    ax.set_xlabel("time")
    ax.set_ylabel("pulse strength")
    ax.set_title(f"CZ Gate Optimized Pulse, Fidelity: {fidelity:.4f}")
    ax.set_ylim([0, 2 * np.pi])
    ax.grid(True, alpha=0.3)

    import io
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    final_img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    img_buffer.close()
    plt.close(fig)

    final_text_for_lucien = f"Final fidelity: {fidelity:.6f}"

    return [
        ImageContent(type="image", data=final_img_base64, mimeType="image/png"),
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
    ... (文档字符串不变) ...
    """
    g = GRAPE_X(num_fourier_terms=fourier_terms)
    avg_fidelities_history_opt = []

    # --- 新增代码: 定义更新频率 ---
    update_interval = 5  # 每 5 次迭代更新一次GUI

    for i in range(iterations):
        avg_fidelity_opt_grid, _ = g.iteration_onestep_numerical_derivative(lr=learning_rate)
        avg_fidelities_history_opt.append(avg_fidelity_opt_grid)

        # --- 新增代码块: 周期性发送实时更新到GUI ---
        if i % update_interval == 0 or i == iterations - 1:
            # 1. 准备实时文本
            live_text = (f"Iteration: {i + 1}/{iterations}\n"
                         f"Current Average Fidelity: {avg_fidelity_opt_grid:.6f}")

            # 2. 生成实时脉冲图像
            plt.close('all')
            fig, ax = plt.subplots(figsize=(10, 6))
            t_plot_live = np.linspace(0, g.t_final, 400)
            phi_plot_live = g.reconstruct_phi_at_t(t_plot_live, g.a_coeffs, g.b_coeffs)
            ax.plot(t_plot_live, phi_plot_live, "r-", linewidth=2, label="Phase φ(t) (Fourier)")
            ax.set_xlabel("time")
            ax.set_ylabel("Phase φ(t) (radians)", color="r")
            ax.tick_params(axis="y", labelcolor="r")
            ax.set_title(f"Optimization in Progress... (Iteration {i + 1})")
            ax.set_ylim([0, 2 * np.pi])
            ax.grid(True, alpha=0.3)
            ax.legend()

            # 3. 将图像转换为base64
            import io
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            live_img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
            img_buffer.close()
            plt.close(fig)

            # 4. 发送到GUI
            send_to_gui(text_content=live_text, images_base64=[live_img_base64])
        # --- 新增代码块结束 ---


    # --- 以下是原始代码的结尾部分，保持不变 ---
    # 它负责生成最终的图像并返回给Lucien
    plt.close('all')
    fig, ax = plt.subplots(figsize=(10, 6))
    t_plot_live = np.linspace(0, g.t_final, 400)
    phi_plot_live = g.reconstruct_phi_at_t(t_plot_live, g.a_coeffs, g.b_coeffs)
    ax.plot(t_plot_live, phi_plot_live, "r-", linewidth=2, label="Phase φ(t) (Fourier)")
    ax.set_xlabel("time")
    ax.set_ylabel("Phase φ(t) (radians)", color="r")
    ax.tick_params(axis="y", labelcolor="r")
    ax.set_title(f"X Gate Optimized Pulse, Avg Fidelity: {g.fidelity:.4f}")
    ax.set_ylim([0, 2 * np.pi])
    ax.grid(True, alpha=0.3)
    ax.legend()

    import io
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    final_img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    img_buffer.close()
    plt.close(fig)
    
    final_text_for_lucien = f"Final average fidelity: {g.fidelity:.6f}"

    return [
        ImageContent(type="image", data=final_img_base64, mimeType="image/png"),
        TextContent(type="text", text=final_text_for_lucien),
    ]


# @mcp.tool(
#     name="generate_circuit_from_stabilizers",
#     description="Generate optimized quantum circuits from stabilizer codes using RL and compare with Qiskit benchmarks.",
# )
# async def generate_circuit_from_stabilizers(
#     stabilizers: Annotated[List[str], "List of stabilizer strings for the quantum error correction code (e.g., ['+ZZ_____', '+_ZZ____', '+XXXXXXX'] for 7-qubit Steane code)"],
#     num_circuits: Annotated[int, "Number of different circuit optimizations to generate"] = 5,
# ) -> list[ImageContent | TextContent]:
#     """
#     Generate optimized quantum circuits from stabilizer generator strings.
    
#     Args:
#         stabilizers: List of stabilizer strings (use '_' for identity, '+'/'-' for sign)
#         num_circuits: Number of different circuit variants to generate
        
#     Returns:
#         List containing text content with circuit information and PNG images of circuit diagrams
#     """
#     try:
#         # Validate stabilizers format
#         if not stabilizers or not isinstance(stabilizers, list):
#             return [TextContent(type="text", text="Error: stabilizers must be a non-empty list of strings")]
        
#         # Convert stabilizers to target tableau
#         try:
#             # Clean stabilizer strings and convert to stim format
#             clean_stabilizers = []
#             for stab in stabilizers:
#                 # Remove leading +/- and convert _ to I
#                 clean_stab = stab.lstrip('+-').replace('_', 'I')
#                 clean_stabilizers.append(clean_stab)
            
#             num_qubits = len(clean_stabilizers[0].lstrip('+-'))
            
#             # Create tableau with proper stabilizer count (allow underconstrained)
#             target_tableau = stim.Tableau.from_stabilizers(
#                 [stim.PauliString(s) for s in clean_stabilizers], 
#                 allow_underconstrained=True
#             )
            
#         except Exception as e:
#             return [TextContent(type="text", text=f"Error parsing stabilizers: {e}")]
        
#         # Setup environment and experiment for circuit generation
#         try:
#             # Load parameters from the same JSON file that minimal_runner.py uses
#             from importlib import resources
#             try:
#                 project_root = resources.files('gate_optimize').parent.parent
#             except (ImportError, AttributeError):
#                 from pathlib import Path
#                 project_root = Path(__file__).resolve().parent.parent.parent.parent
            
#             params_path = str(project_root / 'model' / 'eval' / '7bit_params.json')
            
#             # Import the parse function to load parameters correctly
#             from .circuit.runner import parse
#             args = parse(['-fromjson', params_path])
            
#             # Override specific values for our use case
#             args.qbits = num_qubits  # Use the actual number of qubits from stabilizers
#             utils.set_seed(args.seed)
            
#             # Initialize utils globals required by Environment (matching minimal_runner.py)
#             utils._globals = {
#                 'debug': False,
#                 'dist': args.dist,
#                 'rewardtype': args.rewardtype,
#                 'swanlab': False,
#                 'device': utils.get_device(prefer_cuda=True),
#                 'noise': lambda state: state,
#                 'bufsize': args.bufsize,
#                 'gamma': args.gamma,
#                 'tau': args.tau,
#                 'num_envs': 1  # Override for single environment
#             }
#             utils.args = args
            
#             # Initialize experiment
#             exp = Experiment(args.a, training_req=False, n_workers=1)
            
#             # Generate circuits using the target tableau
#             results = []
#             images = []  # Store images separately
#             results.append(f"QUANTUM CIRCUIT GENERATION FROM STABILIZERS")
#             results.append("=" * 60)
#             results.append(f"Input Stabilizers: {stabilizers}")
#             results.append(f"Number of Qubits: {num_qubits}")
#             results.append(f"Target Tableau Shape: {num_qubits} qubits")
#             results.append(f"Algorithm: {args.a}")
#             results.append(f"Gateset: {args.gateset}")
#             results.append(f"Max Steps: {args.maxsteps}")
#             results.append("")
            
#             # Construct model path using the correct format (matching minimal_runner.py)
#             model_dir = project_root / 'model' / 'plots' / f"{args.qbits}-{args.tol}-{args.name}--{args.exptdate}"
#             model_path = str(model_dir / 'model')
            
#             # Check if model exists for loading later
#             model_exists = os.path.exists(model_path + '.pkl')
#             if model_exists:
#                 results.append(f"✓ Found pre-trained model at {model_path}.pkl")
#             else:
#                 results.append(f"⚠ No pre-trained model found at {model_path}.pkl")
            
#             # Initialize test environment first (required for sample_env)
#             target_state = stim.Tableau(args.qbits)  # Start state
#             env = exp.initialize_test_env(target_state, target_tableau, args.tol, args.maxsteps, args.gateset, args.dist)
#             env.max_steps = int(1 * args.maxsteps)
            
#             # Initialize agent and load model if available (matching minimal_runner.py)
#             model_loaded = False
#             if model_exists and not hasattr(exp, 'agent'):
#                 try:
#                     results.append("🔄 Initializing RL agent...")
#                     if args.a in ['ppo', 'vpg']:
#                         exp.initialize_agent_pg(
#                             policy_hidden=args.phidden,
#                             policy_activ_fn=getattr(torch.nn.functional, args.activfn),
#                             policy_model_max_grad_norm=0.5,
#                             policy_optimizer_fn=lambda net: torch.optim.Adam(net.parameters(), lr=args.plr),
#                             value_hidden=args.vhidden,
#                             value_activ_fn=getattr(torch.nn.functional, args.activfn),
#                             value_model_max_grad_norm=0.5,
#                             value_optimizer_fn=lambda net: torch.optim.Adam(net.parameters(), lr=args.vlr),
#                             entropy_loss_weight=args.entropywt,
#                             gamma=args.gamma,
#                         )
                    
#                     # Load the trained model
#                     results.append("🔄 Loading trained model...")
#                     exp.load_model(model_path)
#                     model_loaded = True
#                     results.append("✅ Successfully loaded pre-trained RL model!")
#                 except Exception as e:
#                     results.append(f"❌ Model loading failed: {e}")
#                     results.append("⚠️ Will use random policy as fallback")
            
#             # Generate multiple circuit variants using evaluate method (like minimal_runner.py)
#             results.append(f"\n🔄 Generating {min(num_circuits, 5)} circuit variants...")
#             try:
#                 best_circuits, _ = exp.evaluate(env, n_eps=min(num_circuits, 5), num_best=min(num_circuits, 5), verbose=0)
                
#                 for circuit_num, (actions, _, _, final_fidelity) in enumerate(best_circuits):
#                     results.append(f"\n--- CIRCUIT VARIANT {circuit_num + 1} ---")
                    
#                     try:
#                         # Convert actions to integers if needed
#                         int_actions = [a.item() if hasattr(a, 'item') else int(a) for a in actions]
                        
#                         # Try to generate circuit diagram using environment method
#                         try:
#                             qc = env.get_inverted_ckt(int_actions)
#                         except (IndexError, AttributeError):
#                             # Fallback: build circuit manually (like minimal_runner.py)
#                             from qiskit import QuantumCircuit
#                             qc = QuantumCircuit(env.qubits)
                            
#                             for action in reversed(int_actions):
#                                 if action < len(env.gates):
#                                     gate_name = env.gates[action]
#                                     # Parse gate name and apply to circuit
#                                     if gate_name.startswith('h('):
#                                         qubit = int(gate_name.split('(')[1].split(')')[0])
#                                         qc.h(qubit)
#                                     elif gate_name.startswith('cnot('):
#                                         qubits = gate_name.split('(')[1].split(')')[0].split(',')
#                                         qc.cx(int(qubits[0]), int(qubits[1]))
#                                     elif gate_name.startswith('s('):
#                                         qubit = int(gate_name.split('(')[1].split(')')[0])
#                                         qc.s(qubit)
#                                     elif gate_name.startswith('sdg('):
#                                         qubit = int(gate_name.split('(')[1].split(')')[0])
#                                         qc.sdg(qubit)
#                                     elif gate_name.startswith('hsdgh('):
#                                         qubit = int(gate_name.split('(')[1].split(')')[0])
#                                         qc.h(qubit)
#                                         qc.sdg(qubit)
#                                         qc.h(qubit)
#                                     elif gate_name.startswith('x('):
#                                         qubit = int(gate_name.split('(')[1].split(')')[0])
#                                         qc.x(qubit)
#                                     elif gate_name.startswith('y('):
#                                         qubit = int(gate_name.split('(')[1].split(')')[0])
#                                         qc.y(qubit)
#                                     elif gate_name.startswith('z('):
#                                         qubit = int(gate_name.split('(')[1].split(')')[0])
#                                         qc.z(qubit)
                        
#                         gate_names = [env.gates[a] for a in int_actions if a < len(env.gates)]
                        
#                         results.append(f"RL Fidelity: {final_fidelity:.6f}")
#                         results.append("RL Circuit Diagram:")
#                         results.append(str(qc.draw('text', fold=80)))  # Convert to string
#                         results.append(f"RL Gate Array: {gate_names}")
#                         results.append(f"RL Gate Count: {len(qc.data)}")
                        
#                         # Generate PNG diagram for RL circuit
#                         try:
#                             plt.close('all')
#                             fig, ax = plt.subplots(figsize=(12, 6))
#                             qc.draw('mpl', ax=ax)
#                             ax.set_title(f"RL Circuit Variant {circuit_num + 1} - Fidelity: {final_fidelity:.6f}, Gate Count = {len(qc.data)}")
                            
#                             import io
#                             img_buffer = io.BytesIO()
#                             fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
#                             img_buffer.seek(0)
#                             img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
#                             img_buffer.close()
#                             plt.close(fig)
                            
#                             # Store image data for later inclusion in results
#                             images.append(ImageContent(type="image", data=img_base64, mimeType="image/png"))
#                         except Exception as img_e:
#                             results.append(f"Note: Could not generate PNG diagram: {img_e}")
                        
#                     except Exception as e:
#                         results.append(f"Error generating circuit diagram: {e}")
#                         results.append(f"Actions taken: {actions}")
#                         results.append(f"Final fidelity: {final_fidelity:.6f}")
                
#                 # Add Qiskit benchmarks for comparison (only once, not per circuit)
#                 try:
#                     import qiskit.quantum_info as qi
#                     import qiskit.synthesis as qs
                    
#                     results.append(f"\n--- QISKIT BENCHMARKS ---")
                    
#                     # Convert to Qiskit format - ensure consistent lengths
#                     stabilizer_strings = [s.lstrip('+-').replace('_', 'I') for s in stabilizers]
                    
#                     # Ensure all stabilizer strings have the same length as num_qubits
#                     max_length = max(len(s) for s in stabilizer_strings)
#                     actual_num_qubits = max(num_qubits, max_length)
                    
#                     # Pad shorter stabilizers with identity operators
#                     padded_stabilizers = []
#                     for s in stabilizer_strings:
#                         if len(s) < actual_num_qubits:
#                             s = s + 'I' * (actual_num_qubits - len(s))
#                         padded_stabilizers.append(s)
                    
#                     cliff = qi.StabilizerState.from_stabilizer_list(padded_stabilizers, allow_underconstrained=True).clifford
                    
#                     # Get benchmark circuits
#                     benchmarks = {
#                         'bravyi': qi.StabilizerState.from_stabilizer_list(padded_stabilizers, allow_underconstrained=True).clifford.to_circuit(),
#                         'ag': qs.synth_clifford_ag(cliff),
#                         'bm': qs.synth_clifford_full(cliff),
#                         'greedy': qs.synth_clifford_greedy(cliff)
#                     }
                    
#                     for method_name, benchmark_qc in benchmarks.items():
#                         results.append(f"\n{method_name.upper()} Circuit:")
#                         results.append(f"Gate Count: {len(benchmark_qc.data)}")
#                         if len(benchmark_qc.data) < 30:  # Only show diagram for small circuits
#                             results.append("Circuit Diagram:")
#                             results.append(str(benchmark_qc.draw('text', fold=80)))
                            
#                             # Generate PNG diagram for benchmark circuit
#                             try:
#                                 plt.close('all')
#                                 fig, ax = plt.subplots(figsize=(12, 6))
#                                 benchmark_qc.draw('mpl', ax=ax)
#                                 ax.set_title(f"{method_name.upper()} Circuit - Gate Count: {len(benchmark_qc.data)}")
                                
#                                 import io
#                                 img_buffer = io.BytesIO()
#                                 fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
#                                 img_buffer.seek(0)
#                                 img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
#                                 img_buffer.close()
#                                 plt.close(fig)
                                
#                                 # Store image data for later inclusion in results
#                                 images.append(ImageContent(type="image", data=img_base64, mimeType="image/png"))
#                             except Exception as img_e:
#                                 results.append(f"Note: Could not generate PNG diagram: {img_e}")
                        
#                 except Exception as e:
#                     results.append(f"Error generating Qiskit benchmarks: {e}")
                    
#             except Exception as e:
#                 results.append(f"Error during circuit evaluation: {e}")
            
#             results.append(f"\n{'=' * 60}")
            
#         except Exception as e:
#             return [TextContent(type="text", text=f"Error during circuit generation: {e}")]
        
#         # Combine text and image results
#         content_list = [TextContent(type="text", text="\n".join(results))]
        
#         # Add circuit diagram images if they were generated
#         if images:
#             content_list.extend(images)
            
#         return content_list
        
        
#     except Exception as e:
#         return [TextContent(type="text", text=f"Unexpected error: {e}")]


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
    
    Args:
        stabilizers: List of stabilizer strings (use '_' for identity, '+'/'-' for sign)
        num_circuits: Number of different circuit variants to generate
        
    Returns:
        List containing text content with circuit information and PNG images of circuit diagrams
    """
    # 初始化变量，确保它们在所有分支中都存在
    results = []
    images = []

    try:
        # 1. 验证输入
        if not stabilizers or not isinstance(stabilizers, list):
            error_text = "Error: stabilizers must be a non-empty list of strings"
            send_to_gui(error_text, [])
            return [TextContent(type="text", text=error_text)]
        
        # 2. 解析稳定子
        try:
            clean_stabilizers = [s.lstrip('+-').replace('_', 'I') for s in stabilizers]
            num_qubits = len(clean_stabilizers[0])
            target_tableau = stim.Tableau.from_stabilizers(
                [stim.PauliString(s) for s in clean_stabilizers], 
                allow_underconstrained=True
            )
        except Exception as e:
            error_text = f"Error parsing stabilizers: {e}"
            send_to_gui(error_text, [])
            return [TextContent(type="text", text=error_text)]
        
        # 3. 设置环境和实验
        try:
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
            
            # --- 开始构建输出 ---
            results.append("QUANTUM CIRCUIT GENERATION FROM STABILIZERS")
            results.append("=" * 60)
            results.append(f"Input Stabilizers: {stabilizers}")
            # ... (其他 results.append 保持不变) ...
            results.append(f"Number of Qubits: {num_qubits}")
            results.append(f"Target Tableau Shape: {num_qubits} qubits")
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

            # 4. 核心逻辑：加载模型并生成线路
            target_state = stim.Tableau(args.qbits)
            env = exp.initialize_test_env(target_state, target_tableau, args.tol, args.maxsteps, args.gateset, args.dist)
            env.max_steps = int(1 * args.maxsteps)
            
            if model_exists and not hasattr(exp, 'agent'):
                # ... (加载模型的 try-except 块保持不变) ...
                try:
                    results.append("🔄 Initializing RL agent...")
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
                    results.append("🔄 Loading trained model...")
                    exp.load_model(model_path)
                    results.append("✅ Successfully loaded pre-trained RL model!")
                except Exception as e:
                    results.append(f"❌ Model loading failed: {e}")
                    results.append("⚠️ Will use random policy as fallback")

            # 5. 生成RL线路和Qiskit基准
            results.append(f"\n🔄 Generating {min(num_circuits, 5)} circuit variants...")
            try:
                best_circuits, _ = exp.evaluate(env, n_eps=min(num_circuits, 5), num_best=min(num_circuits, 5), verbose=0)
                # ... (for 循环处理 best_circuits 并生成图片的代码保持不变) ...
                for circuit_num, (actions, _, _, final_fidelity) in enumerate(best_circuits):
                    results.append(f"\n--- CIRCUIT VARIANT {circuit_num + 1} ---")
                    try:
                        int_actions = [a.item() if hasattr(a, 'item') else int(a) for a in actions]
                        # ... (内部生成 qc 和图片的逻辑) ...
                        try:
                            qc = env.get_inverted_ckt(int_actions)
                        except (IndexError, AttributeError):
                            from qiskit import QuantumCircuit
                            qc = QuantumCircuit(env.qubits)
                            # ... (手动构建 qc 的逻辑) ...
                            for action in reversed(int_actions):
                                if action < len(env.gates):
                                    gate_name = env.gates[action]
                                    # Parse gate name and apply to circuit
                                    if gate_name.startswith('h('):
                                        qubit = int(gate_name.split('(')[1].split(')')[0])
                                        qc.h(qubit)
                                    elif gate_name.startswith('cnot('):
                                        qubits = gate_name.split('(')[1].split(')')[0].split(',')
                                        qc.cx(int(qubits[0]), int(qubits[1]))
                                    elif gate_name.startswith('s('):
                                        qubit = int(gate_name.split('(')[1].split(')')[0])
                                        qc.s(qubit)
                                    elif gate_name.startswith('sdg('):
                                        qubit = int(gate_name.split('(')[1].split(')')[0])
                                        qc.sdg(qubit)
                                    elif gate_name.startswith('hsdgh('):
                                        qubit = int(gate_name.split('(')[1].split(')')[0])
                                        qc.h(qubit)
                                        qc.sdg(qubit)
                                        qc.h(qubit)
                                    elif gate_name.startswith('x('):
                                        qubit = int(gate_name.split('(')[1].split(')')[0])
                                        qc.x(qubit)
                                    elif gate_name.startswith('y('):
                                        qubit = int(gate_name.split('(')[1].split(')')[0])
                                        qc.y(qubit)
                                    elif gate_name.startswith('z('):
                                        qubit = int(gate_name.split('(')[1].split(')')[0])
                                        qc.z(qubit)

                        gate_names = [env.gates[a] for a in int_actions if a < len(env.gates)]
                        results.append(f"RL Fidelity: {final_fidelity:.6f}")
                        # ... (其他 results.append) ...
                        results.append("RL Circuit Diagram:")
                        results.append(str(qc.draw('text', fold=80)))  # Convert to string
                        results.append(f"RL Gate Array: {gate_names}")
                        results.append(f"RL Gate Count: {len(qc.data)}")
                        
                        try:
                            # ... (生成图片并存入 images 列表) ...
                            plt.close('all')
                            fig, ax = plt.subplots(figsize=(12, 6))
                            qc.draw('mpl', ax=ax)
                            ax.set_title(f"RL Circuit Variant {circuit_num + 1} - Fidelity: {final_fidelity:.6f}, Gate Count = {len(qc.data)}")
                            import io
                            img_buffer = io.BytesIO()
                            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                            img_buffer.seek(0)
                            img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
                            img_buffer.close()
                            plt.close(fig)
                            images.append(ImageContent(type="image", data=img_base64, mimeType="image/png"))
                        except Exception as img_e:
                            results.append(f"Note: Could not generate PNG diagram: {img_e}")
                    except Exception as e:
                        results.append(f"Error generating circuit diagram: {e}")

                # ... (Qiskit 基准测试的 try-except 块保持不变) ...
                try:
                    import qiskit.quantum_info as qi
                    import qiskit.synthesis as qs
                    results.append(f"\n--- QISKIT BENCHMARKS ---")
                    # ... (生成Qiskit基准测试并存入 images 列表) ...
                    stabilizer_strings = [s.lstrip('+-').replace('_', 'I') for s in stabilizers]
                    max_length = max(len(s) for s in stabilizer_strings)
                    actual_num_qubits = max(num_qubits, max_length)
                    padded_stabilizers = []
                    for s in stabilizer_strings:
                        if len(s) < actual_num_qubits:
                            s = s + 'I' * (actual_num_qubits - len(s))
                        padded_stabilizers.append(s)
                    cliff = qi.StabilizerState.from_stabilizer_list(padded_stabilizers, allow_underconstrained=True).clifford
                    benchmarks = {
                        'bravyi': qi.StabilizerState.from_stabilizer_list(padded_stabilizers, allow_underconstrained=True).clifford.to_circuit(),
                        'ag': qs.synth_clifford_ag(cliff),
                        'bm': qs.synth_clifford_full(cliff),
                        'greedy': qs.synth_clifford_greedy(cliff)
                    }
                    for method_name, benchmark_qc in benchmarks.items():
                        results.append(f"\n{method_name.upper()} Circuit:")
                        # ... (其他 results.append 和图片生成) ...
                        results.append(f"Gate Count: {len(benchmark_qc.data)}")
                        if len(benchmark_qc.data) < 30:
                            results.append("Circuit Diagram:")
                            results.append(str(benchmark_qc.draw('text', fold=80)))
                            try:
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
                            except Exception as img_e:
                                results.append(f"Note: Could not generate PNG diagram: {img_e}")

                except Exception as e:
                    results.append(f"Error generating Qiskit benchmarks: {e}")

            except Exception as e:
                results.append(f"Error during circuit evaluation: {e}")
            
            results.append(f"\n{'=' * 60}")
        
        except Exception as e:
            # 捕获设置环境时的错误
            error_text = f"Error during setup or circuit generation: {e}"
            send_to_gui(error_text, [])
            return [TextContent(type="text", text=error_text)]
        
        # --- 最终处理逻辑 ---
        # 无论成功与否，到这里 results 和 images 都已经收集完毕
        final_text = "\n".join(results)
        images_base64 = [img.data for img in images]
        
        # 发送到自定义GUI
        send_to_gui(final_text, images_base64)
        
        # 返回给Lucien
        content_list = [TextContent(type="text", text=final_text)]
        if images:
            content_list.extend(images)
            
        return content_list
        
    except Exception as e:
        # 捕获最顶层的未知错误
        error_text = f"An unexpected error occurred: {e}"
        send_to_gui(error_text, [])
        return [TextContent(type="text", text=error_text)]

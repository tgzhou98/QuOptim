
import base64
from typing import Annotated, Literal, List

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

mcp = FastMCP("mcp-gate-optimize")


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
    fig = plt.figure(0)
    for i in range(iterations):
        fidelity = g.iteration_onestep(learning_rate)
        infid.append(abs(1 - fidelity))

    plt.clf()
    time_steps_stair0, pwc_pulse_stair0 = g.PWC_pulse(g.phi)
    plt.plot(time_steps_stair0, pwc_pulse_stair0, "b-")
    plt.xlabel("time")
    plt.ylabel("pulse strength")
    plt.title(f"CZ Gate Optimized Pulse, Fidelity: {fidelity:.4f}")
    plt.ylim([0, 2 * np.pi])

    img_bytes = fig.to_image(format="png", scale=1)
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return [
        ImageContent(type="image", data=img_base64, mimeType="image/png"),
        TextContent(type="text", text=f"Final fidelity: {fidelity:.6f}"),
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

    Args:
        iterations: The number of iterations for the optimization.
        learning_rate: The learning rate for the optimization.
        fourier_terms: The number of Fourier terms to optimize.

    Returns:
        A list containing the plot of the optimized pulses and the final fidelity.
    """
    g = GRAPE_X(
        num_fourier_terms=fourier_terms,
    )
    avg_fidelities_history_opt = []
    fig = plt.figure(0)

    for i in range(iterations):
        avg_fidelity_opt_grid, _ = g.iteration_onestep_numerical_derivative(
            lr=learning_rate
        )
        avg_fidelities_history_opt.append(avg_fidelity_opt_grid)

    ax = fig.add_subplot(1, 1, 1)
    t_plot_live = np.linspace(0, g.t_final, 400)
    phi_plot_live = g.reconstruct_phi_at_t(t_plot_live, g.a_coeffs, g.b_coeffs)
    ax.plot(t_plot_live, phi_plot_live, "r-", label="Phase φ(t) (Fourier)")
    ax.set_ylabel("Phase φ(t) (radians)", color="r")
    ax.tick_params(axis="y", labelcolor="r")
    ax.set_title(f"X Gate Optimized Pulse, Avg Fidelity: {g.fidelity:.4f}")
    ax.set_ylim([0, 2 * np.pi])
    ax.grid(True)

    img_bytes = fig.to_image(format="png", scale=1)
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return [
        ImageContent(type="image", data=img_base64, mimeType="image/png"),
        TextContent(type="text", text=f"Final average fidelity: {g.fidelity:.6f}"),
    ]


@mcp.tool(
    name="generate_circuit_from_stabilizers",
    description="Generate optimized quantum circuits from stabilizer codes using RL and compare with Qiskit benchmarks.",
)
async def generate_circuit_from_stabilizers(
    stabilizers: Annotated[List[str], "List of stabilizer strings for the quantum error correction code (e.g., ['+ZZ_____', '+_ZZ____', '+XXXXXXX'] for 7-qubit Steane code)"],
    num_circuits: Annotated[int, "Number of different circuit optimizations to generate"] = 5,
) -> list[TextContent]:
    """
    Generate optimized quantum circuits from stabilizer generator strings.
    
    Args:
        stabilizers: List of stabilizer strings (use '_' for identity, '+'/'-' for sign)
        num_circuits: Number of different circuit variants to generate
        
    Returns:
        Text content containing circuit diagrams, gate counts, and fidelities
    """
    try:
        # Validate stabilizers format
        if not stabilizers or not isinstance(stabilizers, list):
            return [TextContent(type="text", text="Error: stabilizers must be a non-empty list of strings")]
        
        # Convert stabilizers to target tableau
        try:
            # Clean stabilizer strings and convert to stim format
            clean_stabilizers = []
            for stab in stabilizers:
                # Remove leading +/- and convert _ to I
                clean_stab = stab.lstrip('+-').replace('_', 'I')
                clean_stabilizers.append(clean_stab)
            
            num_qubits = len(clean_stabilizers[0].lstrip('+-'))
            
            # Create tableau with proper stabilizer count (allow underconstrained)
            target_tableau = stim.Tableau.from_stabilizers(
                [stim.PauliString(s) for s in clean_stabilizers], 
                allow_underconstrained=True
            )
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error parsing stabilizers: {e}")]
        
        # Setup environment and experiment for circuit generation
        try:
            # Create temporary arguments object for circuit generation
            class Args:
                def __init__(self):
                    self.qbits = num_qubits
                    self.tol = 0.1
                    self.name = 'stabilizer-circuit'
                    self.exptdate = '20250820'
                    self.seed = 42
                    self.a = 'ppo'  # Use PPO algorithm
            
            args = Args()
            utils.set_seed(args.seed)
            
            # Initialize utils globals required by Environment
            utils._globals['rewardtype'] = 'fidelity'
            utils._globals['debug'] = False
            utils._globals['dist'] = 'clifford'
            utils._globals['device'] = 'cpu'
            utils._globals['bufsize'] = 1000
            utils._globals['gamma'] = 0.99
            utils._globals['tau'] = 0.005
            utils._globals['num_envs'] = 1
            
            # Initialize experiment
            exp = Experiment(args.a, training_req=False, n_workers=1)
            
            # Generate circuits using the target tableau
            results = []
            results.append(f"QUANTUM CIRCUIT GENERATION FROM STABILIZERS")
            results.append("=" * 60)
            results.append(f"Input Stabilizers: {stabilizers}")
            results.append(f"Number of Qubits: {num_qubits}")
            results.append(f"Target Tableau Shape: {num_qubits} qubits")
            results.append("")
            
            # Try to load existing model for circuit generation
            from importlib import resources
            try:
                project_root = resources.files('gate_optimize').parent.parent
            except (ImportError, AttributeError):
                from pathlib import Path
                project_root = Path(__file__).resolve().parent.parent.parent.parent
            
            model_dir = project_root / 'model' / 'plots' / f"{args.qbits}-{args.tol}-final-submission-nopenalty--08-09-2024"
            model_path = str(model_dir / 'model')
            
            if os.path.exists(model_path + '.pkl'):
                try:
                    exp.load_model(model_path)
                    results.append("✓ Loaded pre-trained RL model")
                except:
                    results.append("⚠ Using random policy (no pre-trained model)")
            else:
                results.append("⚠ Using random policy (no pre-trained model found)")
            
            # Generate multiple circuit variants
            for circuit_num in range(min(num_circuits, 5)):  # Limit to 5 for performance
                results.append(f"\n--- CIRCUIT VARIANT {circuit_num + 1} ---")
                
                # Create environment for this target
                env = Environment(
                    num_envs=1,
                    target_state=target_tableau,
                    fidelity_tol=args.tol,
                    max_steps=50,
                    gateset=['cnot', 'h', 's', 'sdg', 'x', 'y', 'z'],  # Standard gateset
                    dist='clifford',  # Use valid distribution
                    seed=args.seed
                )
                
                # Generate circuit using RL agent
                obs = env.reset()
                actions = []
                done = False
                steps = 0
                max_steps = 50  # Prevent infinite loops
                
                while not done and steps < max_steps:
                    try:
                        # Get action from model or random
                        if hasattr(exp, 'policy') and exp.policy is not None:
                            action = exp.get_action(obs, deterministic=True)
                        else:
                            action = np.random.choice(env.action_space)
                        
                        obs, reward, done, info = env.step(action)
                        actions.append(action)
                        steps += 1
                        
                        if done:
                            break
                            
                    except Exception as e:
                        results.append(f"Error during circuit generation: {e}")
                        break
                
                # Get final fidelity and circuit
                final_fidelity = env.curr_fidelity()[0]  # Get first environment's fidelity
                
                try:
                    # Generate circuit diagram
                    qc = env.get_inverted_ckt(actions)
                    gate_names = [env.gates[a] if isinstance(a, int) and a < len(env.gates) 
                                else f"action_{a}" for a in actions]
                    
                    results.append(f"RL Fidelity: {final_fidelity:.6f}")
                    results.append("RL Circuit Diagram:")
                    results.append(str(qc.draw('text', fold=-1)))  # Convert to string
                    results.append(f"RL Gate Array: {gate_names}")
                    results.append(f"RL Gate Count: {len(qc.data)}")
                    
                except Exception as e:
                    results.append(f"Error generating circuit diagram: {e}")
                    results.append(f"Actions taken: {actions}")
                    results.append(f"Final fidelity: {final_fidelity:.6f}")
                
                # Add Qiskit benchmarks for comparison
                try:
                    import qiskit.quantum_info as qi
                    import qiskit.synthesis as qs
                    
                    # Convert to Qiskit format
                    stabilizer_strings = [s.lstrip('+-').replace('_', 'I') for s in stabilizers]
                    cliff = qi.StabilizerState.from_stabilizer_list(stabilizer_strings, allow_underconstrained=True).clifford
                    
                    # Get benchmark circuits
                    benchmarks = {
                        'bravyi': qi.StabilizerState.from_stabilizer_list(stabilizer_strings, allow_underconstrained=True).clifford.to_circuit(),
                        'ag': qs.synth_clifford_ag(cliff),
                        'bm': qs.synth_clifford_full(cliff),
                        'greedy': qs.synth_clifford_greedy(cliff)
                    }
                    
                    results.append(f"\n--- QISKIT BENCHMARKS ---")
                    for method_name, benchmark_qc in benchmarks.items():
                        results.append(f"\n{method_name.upper()} Circuit:")
                        results.append(f"Gate Count: {len(benchmark_qc.data)}")
                        if len(benchmark_qc.data) < 30:  # Only show diagram for small circuits
                            results.append("Circuit Diagram:")
                            results.append(str(benchmark_qc.draw('text', fold=-1)))
                        
                except Exception as e:
                    results.append(f"Error generating Qiskit benchmarks: {e}")
            
            results.append(f"\n{'=' * 60}")
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error during circuit generation: {e}")]
        
        return [TextContent(type="text", text="\n".join(results))]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Unexpected error: {e}")]


@mcp.tool(
    name="generate_steane_code_circuits",
    description="Generate circuits for the 7-qubit Steane quantum error correction code using pre-defined stabilizers.",
)
async def generate_steane_code_circuits(
    num_variants: Annotated[int, "Number of circuit variants to generate"] = 3,
) -> list[TextContent]:
    """
    Generate optimized quantum circuits for the 7-qubit Steane code.
    
    Args:
        num_variants: Number of different circuit optimizations to generate
        
    Returns:
        Text content containing circuit diagrams, gate counts, and fidelities
    """
    # 7-qubit Steane code stabilizers
    steane_stabilizers = [
        '+ZZ_____',
        '+_ZZ____', 
        '+__ZZ___',
        '+___ZZ__',
        '+____ZZ_',
        '+_____ZZ',
        '+XXXXXXX'
    ]
    
    return await generate_circuit_from_stabilizers(steane_stabilizers, num_variants)


# main = mcp.app

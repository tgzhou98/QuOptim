
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
from scipy.linalg import expm
import stim
import tempfile
import os
import torch


mcp = FastMCP("mcp-gate-optimize")


@mcp.tool(
    name="optimize_cz_gate",
    description="Run the GRAPE algorithm to optimize the CZ gate with comprehensive pulse optimization and bidirectional evolution.",
)
async def optimize_cz_gate(
    iterations: Annotated[int, "The number of iterations for the optimization."] = 500,
    learning_rate: Annotated[float, "The learning rate for the optimization."] = 0.5,
    pulse_time: Annotated[float, "Total pulse time duration."] = 7.6,
    omega_max: Annotated[float, "Maximum Rabi frequency."] = 1.0,
    time_steps: Annotated[int, "Number of time steps for pulse discretization."] = 100,
) -> list[ImageContent | TextContent]:
    """
    Run the GRAPE algorithm to optimize the CZ gate with comprehensive analysis.

    Args:
        iterations: The number of iterations for the optimization.
        learning_rate: The learning rate for the optimization.
        pulse_time: Total pulse time duration.
        omega_max: Maximum Rabi frequency.
        time_steps: Number of time steps for pulse discretization.

    Returns:
        A list containing dual-plot visualization and comprehensive optimization results.
    """
    # Initialize GRAPE with custom parameters
    g = GRAPE_CZ()
    
    # Update GRAPE parameters to match input
    g.t_list = np.linspace(0, pulse_time, time_steps + 1)
    g.dt = g.t_list[1] - g.t_list[0]
    g.phi = np.random.rand(len(g.t_list)) * 2 * np.pi
    g.Omega_max = omega_max
    g.HNplus1 = np.array([[1,0,0,0],[0,0,0,0],[0,0,2,0],[0,0,0,0]]) / g.dt
    
    print("Initialized CZ gate GRAPE optimization with bidirectional evolution.")
    
    # Track optimization progress
    progress_text = []
    progress_text.append("=== CZ GATE GRAPE OPTIMIZATION ===")
    progress_text.append("Pulse Parameters:")
    progress_text.append(f"  - Total time: {pulse_time:.3f}")
    progress_text.append(f"  - Time steps: {time_steps}")
    progress_text.append(f"  - dt: {g.dt:.6f}")
    progress_text.append(f"  - Omega_max: {omega_max}")
    progress_text.append(f"  - Learning rate: {learning_rate}")
    progress_text.append(f"  - Iterations: {iterations}")
    progress_text.append("")
    
    progress_text.append("Initial Conditions:")
    progress_text.append("  - Initial state: |01⟩ (01, 0r, 11, W) - unnormalized")
    progress_text.append("  - Target state: |01⟩ → |01⟩, |11⟩ → -|11⟩ (CZ gate)")
    progress_text.append("  - Basis states: 4D Hilbert space")
    progress_text.append(f"  - Initial random phi range: [0, 2π]")
    progress_text.append("")
    
    infid = []
    fidelity_history = []
    
    # Optimization loop with progress tracking
    progress_text.append("Optimization Progress:")
    for i in range(iterations):
        fidelity = g.iteration_onestep(learning_rate)
        infid.append(abs(1 - fidelity))
        fidelity_history.append(fidelity)
        
        # Report progress at key intervals
        interval = max(1, iterations // 10)  # Avoid division by zero for small iterations
        if i % interval == 0 or i == iterations - 1:
            progress_text.append(f"  Iter {i:4d}: Fidelity = {fidelity:.6f}")
    
    progress_text.append("")
    progress_text.append("Final Results:")
    progress_text.append(f"  - Final fidelity: {fidelity:.6f}")
    progress_text.append(f"  - Final infidelity: {abs(1 - fidelity):.2e}")
    progress_text.append(f"  - Pulse optimization range: [0, 2π]")
    progress_text.append(f"  - Phase values range: [{g.phi.min():.3f}, {g.phi.max():.3f}]")
    
    # Bidirectional evolution analysis
    try:
        A_list, B_list = g.BidirectEvolution()
        progress_text.append("")
        progress_text.append("Bidirectional Evolution Analysis:")
        progress_text.append(f"  - Forward evolution matrices (B_j): {len(B_list)} steps")
        progress_text.append(f"  - Backward evolution matrices (A_j): {len(A_list)} steps")
        progress_text.append("  - Evolution: U_total = U_N...U_1 with intermediate states")
    except Exception as e:
        progress_text.append(f"  - Bidirectional evolution analysis failed: {e}")

    # Create dual-plot visualization
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Optimized pulse shape
    time_steps_stair, pwc_pulse_stair = g.PWC_pulse(g.phi)
    ax1.plot(time_steps_stair, pwc_pulse_stair, "b-", linewidth=2, label='φ(t) - Optimized Phase')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Phase φ(t)")
    ax1.set_title(f"CZ Gate Optimized Pulse Shape | Final Fidelity: {fidelity:.6f}")
    ax1.set_ylim([0, 2 * np.pi])
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add π markers on y-axis
    ax1.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax1.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    # Plot 2: Optimization convergence (both fidelity and infidelity)
    ax2_twin = ax2.twinx()
    
    # Plot fidelity on left axis
    ax2.plot(fidelity_history, 'g-', linewidth=2, label='Fidelity')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Fidelity', color='g')
    ax2.set_title('CZ Gate Optimization Convergence')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])
    ax2.tick_params(axis='y', labelcolor='g')
    
    # Plot infidelity on right axis (log scale)
    ax2_twin.semilogy(infid, 'r-', linewidth=2, label='Infidelity (log)')
    ax2_twin.set_ylabel('Infidelity (log scale)', color='r')
    ax2_twin.tick_params(axis='y', labelcolor='r')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Add final value annotation
    ax2.axhline(y=fidelity, color='g', linestyle='--', alpha=0.7)
    ax2.text(iterations*0.7, fidelity + 0.05, 
            f'Final: {fidelity:.6f}', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

    plt.tight_layout()

    # Convert figure to bytes
    import io
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    img_buffer.close()
    plt.close(fig)

    return [
        ImageContent(type="image", data=img_base64, mimeType="image/png"),
        TextContent(type="text", text="\n".join(progress_text)),
    ]


@mcp.tool(
    name="optimize_x_gate",
    description="Run the GRAPE algorithm to optimize the X gate with Fourier coefficients and comprehensive pulse shaping.",
)
async def optimize_x_gate(
    iterations: Annotated[int, "The number of iterations for the optimization."] = 500,
    learning_rate: Annotated[float, "The learning rate for the optimization."] = 0.05,
    fourier_terms: Annotated[int, "The number of Fourier terms to optimize."] = 6,
    pulse_time: Annotated[float, "Total pulse time (in units of π)."] = 2.5,
    omega_max: Annotated[float, "Maximum Rabi frequency."] = 1.0,
    rise_fall_ratio: Annotated[float, "Rise/fall time as fraction of total pulse time."] = 0.05,
    optimize_sine_terms: Annotated[bool, "Whether to optimize sine terms (for asymmetric pulses)."] = False,
    num_slices_for_time_evolution: Annotated[int, "Number of time slices for evolution accuracy."] = 50,
) -> list[ImageContent | TextContent]:
    """
    Run the GRAPE algorithm to optimize the X gate with Fourier coefficients.

    Args:
        iterations: The number of iterations for the optimization.
        learning_rate: The learning rate for the optimization.
        fourier_terms: The number of Fourier terms to optimize.
        pulse_time: Total pulse time in units of π.
        omega_max: Maximum Rabi frequency.
        rise_fall_ratio: Rise/fall time as fraction of total pulse time.
        optimize_sine_terms: Whether to optimize sine terms for asymmetric pulses.
        num_slices_for_time_evolution: Number of time slices for evolution accuracy.

    Returns:
        A list containing comprehensive pulse optimization plots and results.
    """
    # Calculate derived parameters matching optimize_x.py
    total_pulse_time_L = np.pi * pulse_time
    rise_t = total_pulse_time_L * rise_fall_ratio
    fall_t = total_pulse_time_L * rise_fall_ratio
    
    # Initialize GRAPE_X with comprehensive parameters matching main function
    g = GRAPE_X(
        t_final=total_pulse_time_L,
        N_slices_for_evolution=num_slices_for_time_evolution,
        num_fourier_terms=fourier_terms,
        omega_max_val=omega_max,
        rise_time=rise_t,
        fall_time=fall_t,
        initial_coeffs_file=None,
        optimize_sine_terms=optimize_sine_terms
    )
    
    # Track optimization history
    avg_fidelities_history_opt = []
    finite_diff_coeff_step = 1e-5  # Matching optimize_x.py
    
    # Create progress text for detailed reporting
    progress_text = []
    progress_text.append("=== X GATE FOURIER OPTIMIZATION ===")
    progress_text.append(f"Pulse Parameters:")
    progress_text.append(f"  - Total time: {total_pulse_time_L:.3f} (π × {pulse_time})")
    progress_text.append(f"  - Fourier terms: {fourier_terms}")
    progress_text.append(f"  - Evolution slices: {num_slices_for_time_evolution}")
    progress_text.append(f"  - Rise/fall times: {rise_t:.3f}")
    progress_text.append(f"  - Optimize sine terms: {optimize_sine_terms}")
    progress_text.append(f"  - Learning rate: {learning_rate}")
    progress_text.append(f"  - Iterations: {iterations}")
    progress_text.append("")
    
    # Initial conditions reporting
    progress_text.append("Initial Conditions:")
    progress_text.append(f"  - Initial state: |0⟩")
    progress_text.append(f"  - Target state: |1⟩ (X gate)")
    progress_text.append(f"  - Noise conditions: {len(g.noise_conditions)} robustness points")
    progress_text.append(f"  - Initial a_coeffs[0] (DC): {g.a_coeffs[0]:.4f}")
    progress_text.append("")
    
    # Run optimization with detailed progress tracking
    progress_text.append("Optimization Progress:")
    for i in range(iterations):
        avg_fidelity_opt_grid, current_fids_dict = g.iteration_onestep_numerical_derivative(
            lr=learning_rate, d_coeff=finite_diff_coeff_step
        )
        avg_fidelities_history_opt.append(avg_fidelity_opt_grid)
        
        # Log progress at key intervals
        if i == 0 or i == 9 or i % 50 == 0 or i == iterations - 1:
            progress_text.append(f"  Iter {i:4d}: Avg Fidelity = {avg_fidelity_opt_grid:.6f}")
    
    progress_text.append("")
    progress_text.append("Final Results:")
    progress_text.append(f"  - Final average fidelity: {g.fidelity:.6f}")
    progress_text.append(f"  - Optimized a_coeffs[0] (DC): {g.a_coeffs[0]:.4f}")
    progress_text.append(f"  - Fourier coeffs range: a[1:] ∈ [{np.min(g.a_coeffs[1:]):.4f}, {np.max(g.a_coeffs[1:]):.4f}]")
    if optimize_sine_terms:
        progress_text.append(f"  - Sine coeffs range: b[1:] ∈ [{np.min(g.b_coeffs[1:]):.4f}, {np.max(g.b_coeffs[1:]):.4f}]")
    
    # Add individual fidelities for robustness analysis (3x3 optimization grid)
    progress_text.append("")
    progress_text.append("Robustness Analysis (3x3 Optimization Grid):")
    final_fidelity, final_individual_fids = g.calculate_average_fidelity_fourier(
        g.a_coeffs, g.b_coeffs, g.noise_conditions
    )
    for condition in g.noise_conditions:
        fid_val = final_individual_fids[condition['name']]
        progress_text.append(f"  {condition['name']}: {fid_val:.6f}")

    # Add comprehensive 11x11 robustness evaluation (like in optimize_x.py)
    progress_text.append("")
    progress_text.append("Extended Robustness Analysis (11x11 Evaluation Grid):")
    
    # Create 11x11 evaluation grid
    rabi_factors_eval = np.linspace(0.95, 1.05, 11)
    detuning_multipliers_eval = np.linspace(-0.05, 0.05, 11)
    fidelities_eval_grid = np.zeros((len(detuning_multipliers_eval), len(rabi_factors_eval)))
    
    eval_conditions_list = [
        {'name': f'Eval_R{r:.3f}_D{d:.3f}', 'omega_factor': r, 'detuning_val': d * g.Omega_max}
        for d in detuning_multipliers_eval for r in rabi_factors_eval
    ]
    
    all_eval_fids = {}
    for condition in eval_conditions_list:
        fid_val, _ = g.calculate_average_fidelity_fourier(g.a_coeffs, g.b_coeffs, [condition])
        all_eval_fids[condition['name']] = fid_val
    
    for d_idx, d_multiplier in enumerate(detuning_multipliers_eval):
        for r_idx, r_factor in enumerate(rabi_factors_eval):
            cond_name = f'Eval_R{r_factor:.3f}_D{d_multiplier:.3f}'
            fidelities_eval_grid[d_idx, r_idx] = all_eval_fids.get(cond_name, 0.0)
    
    avg_eval_fidelity = np.mean(fidelities_eval_grid)
    min_eval_fidelity = np.min(fidelities_eval_grid)
    max_eval_fidelity = np.max(fidelities_eval_grid)
    
    progress_text.append(f"  - Average evaluation fidelity (11x11): {avg_eval_fidelity:.6f}")
    progress_text.append(f"  - Minimum evaluation fidelity: {min_eval_fidelity:.6f}")
    progress_text.append(f"  - Maximum evaluation fidelity: {max_eval_fidelity:.6f}")
    progress_text.append(f"  - Robustness range: {max_eval_fidelity - min_eval_fidelity:.6f}")
    
    # Add Bloch sphere trajectory analysis
    progress_text.append("")
    progress_text.append("Enhanced Bloch Sphere Trajectory Analysis:")
    progress_text.append("  - Simulated quantum state evolution on Bloch sphere")
    progress_text.append("  - Ideal trajectory: Perfect Ω_max, zero detuning")
    progress_text.append("  - Error case 1: Ω_max +5% (Rabi frequency variation)")
    progress_text.append("  - Error case 2: Ω_max -5% (Rabi frequency variation)")  
    progress_text.append("  - Error case 3: Δ = +5% Ω_0 (positive detuning error)")
    progress_text.append("  - Error case 4: Δ = -5% Ω_0 (negative detuning error)")
    progress_text.append("  - All trajectories start from |0⟩ state and evolve to target |1⟩")

    # Create comprehensive visualization (4 subplots in 2x2 grid: pulse, convergence, robustness map, Bloch sphere)
    plt.close('all')
    fig = plt.figure(figsize=(16, 12))
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2) 
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4, projection='3d')
    
    # Plot 1: Optimized pulse with amplitude envelope
    t_plot_live = np.linspace(0, g.t_final, 400)
    phi_plot_live = g.reconstruct_phi_at_t(t_plot_live, g.a_coeffs, g.b_coeffs)
    ax1.plot(t_plot_live, phi_plot_live, 'r-', linewidth=2, label='Phase φ(t) (Fourier)')
    ax1.set_xlabel(f'Time (L={g.t_final:.2f})')
    ax1.set_ylabel('Phase φ(t) (radians)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.set_title(f'Optimized X Gate Pulse (N_terms={fourier_terms})\nFinal Fidelity: {g.fidelity:.6f}')
    ax1.set_ylim([0, 2 * np.pi])
    ax1.grid(True, alpha=0.3)
    
    # Add amplitude envelope overlay
    ax1_twin = ax1.twinx()
    time_steps_amp, amp_stair = g.PWC_pulse(g.omega_envelope / omega_max)
    ax1_twin.plot(time_steps_amp, amp_stair, 'g:', alpha=0.7, label='Amplitude Ω(t)/Ω_max')
    ax1_twin.set_ylabel('Normalized Amplitude', color='g')
    ax1_twin.tick_params(axis='y', labelcolor='g')
    ax1_twin.set_ylim([-0.05, 1.1])
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    # Plot 2: Optimization convergence
    ax2.plot(avg_fidelities_history_opt, 'k-', linewidth=2, label='Average Fidelity (3×3 Robustness Grid)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Average Fidelity')
    ax2.set_title('Optimization Convergence')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim([0, 1.1])
    
    # Add final value annotation
    if avg_fidelities_history_opt:
        ax2.axhline(y=avg_fidelities_history_opt[-1], color='r', linestyle='--', alpha=0.7)
        ax2.text(len(avg_fidelities_history_opt)*0.7, avg_fidelities_history_opt[-1] + 0.05, 
                f'Final: {avg_fidelities_history_opt[-1]:.6f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    # Plot 3: Robustness heatmap (11x11 evaluation grid)
    im = ax3.imshow(fidelities_eval_grid, 
                   extent=[rabi_factors_eval[0], rabi_factors_eval[-1], 
                          detuning_multipliers_eval[0], detuning_multipliers_eval[-1]], 
                   origin='lower', aspect='auto', cmap='viridis', vmin=0.7, vmax=1.0)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax3, label='Fidelity')
    
    ax3.set_xlabel('Rabi Frequency Factor (Ω/Ω_ideal)')
    ax3.set_ylabel('Detuning Multiplier (Δ/Ω_max)')
    ax3.set_title(f'Robustness Map (11x11 Eval Grid)\nAvg. Eval Fidelity: {avg_eval_fidelity:.6f}')
    
    # Mark optimization grid points (3x3)
    opt_marker_rabis_plot = g.rabi_factors_opt
    opt_marker_detunings_plot = g.detuning_multipliers_opt
    ax3.scatter(np.tile(opt_marker_rabis_plot, len(opt_marker_detunings_plot)), 
               np.repeat(opt_marker_detunings_plot, len(opt_marker_rabis_plot)),
               s=50, facecolors='none', edgecolors='red', marker='o', label='3x3 Opt. Grid Points')
    ax3.legend(fontsize='small', loc='lower left')
    ax3.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    ax3.set_xticks(rabi_factors_eval[::2])
    ax3.set_yticks(detuning_multipliers_eval[::2])

    # Plot 4: Bloch sphere trajectory analysis (inspired by analyze_Fourier_Pulse.py)
    def state_to_bloch_vector(psi):
        """Convert quantum state to Bloch sphere coordinates."""
        psi = psi / np.linalg.norm(psi)
        sx = (psi.conj().T @ g.sigmax @ psi)[0,0]
        sy = (psi.conj().T @ g.sigmay @ psi)[0,0]
        sz = (psi.conj().T @ g.sigmaz @ psi)[0,0]
        return np.real(sx), np.real(sy), np.real(sz)
    
    def simulate_trajectory(initial_psi, a_c, b_c, omega_factor, detuning_val, n_steps=50):
        """Simulate quantum evolution trajectory on Bloch sphere."""
        psi_current = initial_psi.copy()
        bloch_history = [state_to_bloch_vector(psi_current)]
        dt_sim = g.t_final / n_steps
        
        for k in range(n_steps):
            t_mid = (k + 0.5) * dt_sim
            phi_at_t = g.reconstruct_phi_at_t(np.array([t_mid]), a_c, b_c)[0]
            
            # Get Hamiltonian at this time point with noise
            H_control = omega_factor * g.Omega_max * (np.cos(phi_at_t) * g.H1 - np.sin(phi_at_t) * g.H2)
            H_detuning = detuning_val * g.H_detuning_term
            H_total = H_control + H_detuning
            
            U_step = expm(-1j * H_total * dt_sim)
            psi_current = U_step @ psi_current
            bloch_history.append(state_to_bloch_vector(psi_current))
        
        return bloch_history
    
    # Simulate trajectories for ideal and 4 error cases (like analyze_Fourier_Pulse.py)
    initial_psi = g.initial_state.copy()
    error_factor = 0.05  # 5% error
    
    # Ideal trajectory
    bloch_ideal = simulate_trajectory(initial_psi, g.a_coeffs, g.b_coeffs, 1.0, 0.0)
    
    # Define 4 error cases (matching analyze_Fourier_Pulse.py)
    error_cases = [
        {"label": "Ω_max +5%", "omega_factor": 1 + error_factor, "detuning_val": 0.0, "color": "red", "style": "--"},
        {"label": "Ω_max -5%", "omega_factor": 1 - error_factor, "detuning_val": 0.0, "color": "orange", "style": "-."},
        {"label": "Δ = +5% Ω_0", "omega_factor": 1.0, "detuning_val": error_factor * g.Omega_max, "color": "green", "style": ":"},
        {"label": "Δ = -5% Ω_0", "omega_factor": 1.0, "detuning_val": -error_factor * g.Omega_max, "color": "purple", "style": "-"}
    ]
    
    # Simulate all error trajectories
    error_trajectories = []
    for case in error_cases:
        bloch_error = simulate_trajectory(initial_psi, g.a_coeffs, g.b_coeffs, 
                                        case["omega_factor"], case["detuning_val"])
        error_trajectories.append((case, bloch_error))
    
    # Draw Bloch sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax4.plot_surface(x_sphere, y_sphere, z_sphere, color='lightblue', alpha=0.05, linewidth=0)
    
    # Plot ideal trajectory
    bx_ideal, by_ideal, bz_ideal = zip(*bloch_ideal)
    ax4.plot(bx_ideal, by_ideal, bz_ideal, 'b-', linewidth=2.5, label='Ideal Trajectory', alpha=0.9)
    ax4.scatter(bx_ideal[0], by_ideal[0], bz_ideal[0], color='blue', s=60, marker='o')  # Start point
    ax4.scatter(bx_ideal[-1], by_ideal[-1], bz_ideal[-1], color='blue', s=80, marker='X')  # End point
    
    # Plot all 4 error trajectories
    for i, (case, bloch_error) in enumerate(error_trajectories):
        bx_error, by_error, bz_error = zip(*bloch_error)
        ax4.plot(bx_error, by_error, bz_error, 
                color=case["color"], linestyle=case["style"], linewidth=1.8, 
                label=f'{case["label"]} Trajectory', alpha=0.8)
        # End point for each error case
        markers = ['P', 's', '^', 'D']  # Different markers for each error case
        ax4.scatter(bx_error[-1], by_error[-1], bz_error[-1], 
                   color=case["color"], s=70, marker=markers[i])
    
    ax4.set_xlabel('⟨σx⟩')
    ax4.set_ylabel('⟨σy⟩')
    ax4.set_zlabel('⟨σz⟩')
    ax4.set_xlim([-1.1, 1.1])
    ax4.set_ylim([-1.1, 1.1])
    ax4.set_zlim([-1.1, 1.1])
    ax4.set_title('Bloch Sphere Trajectories\nIdeal + 4 Error Cases')
    ax4.legend(fontsize='x-small', loc='upper left', bbox_to_anchor=(0.02, 0.98))
    ax4.view_init(elev=20, azim=45)

    plt.tight_layout()

    # Convert figure to bytes
    import io
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    img_buffer.close()
    plt.close(fig)

    return [
        ImageContent(type="image", data=img_base64, mimeType="image/png"),
        TextContent(type="text", text="\n".join(progress_text)),
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
    
    Args:
        stabilizers: List of stabilizer strings (use '_' for identity, '+'/'-' for sign)
        num_circuits: Number of different circuit variants to generate
        
    Returns:
        List containing text content with circuit information and PNG images of circuit diagrams
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
            # Load parameters from the same JSON file that minimal_runner.py uses
            from importlib import resources
            try:
                project_root = resources.files('gate_optimize').parent.parent
            except (ImportError, AttributeError):
                from pathlib import Path
                project_root = Path(__file__).resolve().parent.parent.parent.parent
            
            params_path = str(project_root / 'model' / 'eval' / '7bit_params.json')
            
            # Import the parse function to load parameters correctly
            from .circuit.runner import parse
            args = parse(['-fromjson', params_path])
            
            # Override specific values for our use case
            args.qbits = num_qubits  # Use the actual number of qubits from stabilizers
            utils.set_seed(args.seed)
            
            # Initialize utils globals required by Environment (matching minimal_runner.py)
            utils._globals = {
                'debug': False,
                'dist': args.dist,
                'rewardtype': args.rewardtype,
                'swanlab': False,
                'device': utils.get_device(prefer_cuda=True),
                'noise': lambda state: state,
                'bufsize': args.bufsize,
                'gamma': args.gamma,
                'tau': args.tau,
                'num_envs': 1  # Override for single environment
            }
            utils.args = args
            
            # Initialize experiment
            exp = Experiment(args.a, training_req=False, n_workers=1)
            
            # Generate circuits using the target tableau
            results = []
            images = []  # Store images separately
            results.append(f"QUANTUM CIRCUIT GENERATION FROM STABILIZERS")
            results.append("=" * 60)
            results.append(f"Input Stabilizers: {stabilizers}")
            results.append(f"Number of Qubits: {num_qubits}")
            results.append(f"Target Tableau Shape: {num_qubits} qubits")
            results.append(f"Algorithm: {args.a}")
            results.append(f"Gateset: {args.gateset}")
            results.append(f"Max Steps: {args.maxsteps}")
            results.append("")
            
            # Construct model path using the correct format (matching minimal_runner.py)
            model_dir = project_root / 'model' / 'plots' / f"{args.qbits}-{args.tol}-{args.name}--{args.exptdate}"
            model_path = str(model_dir / 'model')
            
            # Check if model exists for loading later
            model_exists = os.path.exists(model_path + '.pkl')
            if model_exists:
                results.append(f"✓ Found pre-trained model at {model_path}.pkl")
            else:
                results.append(f"⚠ No pre-trained model found at {model_path}.pkl")
            
            # Initialize test environment first (required for sample_env)
            target_state = stim.Tableau(args.qbits)  # Start state
            env = exp.initialize_test_env(target_state, target_tableau, args.tol, args.maxsteps, args.gateset, args.dist)
            env.max_steps = int(1 * args.maxsteps)
            
            # Initialize agent and load model if available (matching minimal_runner.py)
            model_loaded = False
            if model_exists and not hasattr(exp, 'agent'):
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
                    
                    # Load the trained model
                    results.append("🔄 Loading trained model...")
                    exp.load_model(model_path)
                    model_loaded = True
                    results.append("✅ Successfully loaded pre-trained RL model!")
                except Exception as e:
                    results.append(f"❌ Model loading failed: {e}")
                    results.append("⚠️ Will use random policy as fallback")
            
            # Generate multiple circuit variants using evaluate method (like minimal_runner.py)
            results.append(f"\n🔄 Generating {min(num_circuits, 5)} circuit variants...")
            try:
                best_circuits, _ = exp.evaluate(env, n_eps=min(num_circuits, 5), num_best=min(num_circuits, 5), verbose=0)
                
                for circuit_num, (actions, _, _, final_fidelity) in enumerate(best_circuits):
                    results.append(f"\n--- CIRCUIT VARIANT {circuit_num + 1} ---")
                    
                    try:
                        # Convert actions to integers if needed
                        int_actions = [a.item() if hasattr(a, 'item') else int(a) for a in actions]
                        
                        # Try to generate circuit diagram using environment method

                        qc = env.get_inverted_ckt(int_actions)
                        
                        gate_names = [env.gates[a] for a in int_actions if a < len(env.gates)]
                        
                        results.append(f"RL Fidelity: {final_fidelity:.6f}")
                        results.append("RL Circuit Diagram:")
                        results.append(str(qc.draw('text', fold=80)))  # Convert to string
                        results.append(f"RL Gate Array: {gate_names}")
                        results.append(f"RL Gate Count: {len(qc.data)}")
                        
                        # Generate PNG diagram for RL circuit
                        try:
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
                            
                            # Store image data for later inclusion in results
                            images.append(ImageContent(type="image", data=img_base64, mimeType="image/png"))
                        except Exception as img_e:
                            results.append(f"Note: Could not generate PNG diagram: {img_e}")
                        
                    except Exception as e:
                        results.append(f"Error generating circuit diagram: {e}")
                        results.append(f"Actions taken: {actions}")
                        results.append(f"Final fidelity: {final_fidelity:.6f}")
                
                # Add Qiskit benchmarks for comparison (only once, not per circuit)
                try:
                    import qiskit.quantum_info as qi
                    import qiskit.synthesis as qs
                    
                    results.append(f"\n--- QISKIT BENCHMARKS ---")
                    
                    # Convert to Qiskit format - ensure consistent lengths
                    stabilizer_strings = [s.lstrip('+-').replace('_', 'I') for s in stabilizers]
                    
                    # Ensure all stabilizer strings have the same length as num_qubits
                    max_length = max(len(s) for s in stabilizer_strings)
                    actual_num_qubits = max(num_qubits, max_length)
                    
                    # Pad shorter stabilizers with identity operators
                    padded_stabilizers = []
                    for s in stabilizer_strings:
                        if len(s) < actual_num_qubits:
                            s = s + 'I' * (actual_num_qubits - len(s))
                        padded_stabilizers.append(s)
                    
                    cliff = qi.StabilizerState.from_stabilizer_list(padded_stabilizers, allow_underconstrained=True).clifford
                    
                    # Get benchmark circuits
                    benchmarks = {
                        'bravyi': qi.StabilizerState.from_stabilizer_list(padded_stabilizers, allow_underconstrained=True).clifford.to_circuit(),
                        'ag': qs.synth_clifford_ag(cliff),
                        'bm': qs.synth_clifford_full(cliff),
                        'greedy': qs.synth_clifford_greedy(cliff)
                    }
                    
                    for method_name, benchmark_qc in benchmarks.items():
                        results.append(f"\n{method_name.upper()} Circuit:")
                        results.append(f"Gate Count: {len(benchmark_qc.data)}")
                        if len(benchmark_qc.data) < 30:  # Only show diagram for small circuits
                            results.append("Circuit Diagram:")
                            results.append(str(benchmark_qc.draw('text', fold=80)))
                            
                            # Generate PNG diagram for benchmark circuit
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
                                
                                # Store image data for later inclusion in results
                                images.append(ImageContent(type="image", data=img_base64, mimeType="image/png"))
                            except Exception as img_e:
                                results.append(f"Note: Could not generate PNG diagram: {img_e}")
                        
                except Exception as e:
                    results.append(f"Error generating Qiskit benchmarks: {e}")
                    
            except Exception as e:
                results.append(f"Error during circuit evaluation: {e}")
            
            results.append(f"\n{'=' * 60}")
            
        except Exception as e:
            return [TextContent(type="text", text=f"Error during circuit generation: {e}")]
        
        # Combine text and image results
        content_list = [TextContent(type="text", text="\n".join(results))]
        
        # Add circuit diagram images if they were generated
        if images:
            content_list.extend(images)
            
        return content_list
        
    except Exception as e:
        return [TextContent(type="text", text=f"Unexpected error: {e}")]




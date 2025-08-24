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
from .circuit.runner import parse
from .circuit.fidelity_simulation import simulate_circuit_fidelity
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import expm
import stim
import tempfile
import os
import torch
import base64

from qiskit import QuantumCircuit
from qiskit import qasm3
import qiskit.quantum_info as qi
import qiskit.synthesis as qs
from qiskit.compiler import transpile

import numpy as np
from matplotlib import pyplot as plt
from importlib import resources

from scipy.optimize import curve_fit

import io

from .circuit.fidelity_simulation import simulate_circuit_fidelity_custom


mcp = FastMCP("mcp-gate-optimize")

GUI_ENDPOINT = "http://127.0.0.1:12345/update"

def send_to_gui(**kwargs):
    try:
        kwargs.setdefault("status", "finished")
        requests.post(GUI_ENDPOINT, json=kwargs, timeout=2)
    except requests.exceptions.RequestException:
        pass


@mcp.tool(
    name="optimize_cz_gate",
    description="Run the GRAPE algorithm to optimize the CZ gate with comprehensive pulse optimization and bidirectional evolution.",
)
async def optimize_cz_gate(
    iterations: Annotated[int, "The number of iterations for the optimization."] = 500,
    learning_rate: Annotated[float, "The learning rate for the optimization."] = 0.5,
    time_steps: Annotated[int, "Number of time steps for pulse discretization."] = 100,
) -> list[ImageContent | TextContent]:
    """
    Run the GRAPE algorithm to optimize the CZ gate with comprehensive analysis.

    Args:
        iterations: The number of iterations for the optimization.
        learning_rate: The learning rate for the optimization.
        time_steps: Number of time steps for pulse discretization.

    Returns:
        A list containing dual-plot visualization and comprehensive optimization results.
    """
    # Set default values inside the function
    pulse_time = 7.6
    omega_max = 1.0
    
    # Initialize GRAPE with custom parameters
    g = GRAPE_CZ()
    
    # Update GRAPE parameters to match input
    g.t_list = np.linspace(0, pulse_time, time_steps + 1)
    g.dt = g.t_list[1] - g.t_list[0]
    g.phi = np.random.rand(len(g.t_list)) * 2 * np.pi
    g.Omega_max = omega_max
    g.HNplus1 = np.array([[1,0,0,0],[0,0,0,0],[0,0,2,0],[0,0,0,0]]) / g.dt
    
    # print("Initialized CZ gate GRAPE optimization with bidirectional evolution.")
    
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
    # Live image for GUI

    fidelity_history = []
    update_interval = 10
    live_img_base64 = ""
    fidelity = 0.0

    progress_text.append("Optimization Progress:")
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

            infid.append(abs(1 - fidelity))
            fidelity_history.append(fidelity)
            
            # Report progress at key intervals
            interval = max(1, iterations // 10)  # Avoid division by zero for small iterations
            if i % interval == 0 or i == iterations - 1:
                progress_text.append(f"  Iter {i:4d}: Fidelity = {fidelity:.6f}")
            
    final_text_for_lucien = f"Final fidelity: {fidelity:.6f}"
    
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
    ax2.text(0.98, 0.02, f'Final: {fidelity:.6f}', 
            transform=ax2.transAxes, 
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

    plt.tight_layout()

    # Convert figure to bytes for the final MCP response
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
    num_slices_for_time_evolution: Annotated[int, "Number of time slices for evolution accuracy."] = 50,
) -> list[ImageContent | TextContent]:
    """
    Run the GRAPE algorithm to optimize the X gate with Fourier coefficients.

    Args:
        iterations: The number of iterations for the optimization.
        learning_rate: The learning rate for the optimization.
        fourier_terms: The number of Fourier terms to optimize.
        num_slices_for_time_evolution: Number of time slices for evolution accuracy.

    Returns:
        A list containing comprehensive pulse optimization plots and results.
    """
    # Set default values inside the function
    pulse_time = 2.5
    omega_max = 1.0
    rise_fall_ratio = 0.05
    optimize_sine_terms = False
    
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
    


    fidelity_history = []
    update_interval = 5
    live_img_base64 = ""
    avg_fidelity_opt_grid = 0.0

    # Run optimization with detailed progress tracking
    progress_text.append("Optimization Progress:")
    for i in range(iterations):
        avg_fidelity_opt_grid, _ = g.iteration_onestep_numerical_derivative(
            lr=learning_rate, d_coeff=finite_diff_coeff_step
        )
        avg_fidelities_history_opt.append(avg_fidelity_opt_grid)
        
        # Log progress at key intervals
        if i == 0 or i == 9 or i % 50 == 0 or i == iterations - 1:
            progress_text.append(f"  Iter {i:4d}: Avg Fidelity = {avg_fidelity_opt_grid:.6f}")
        
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
    name="generate_circuits",
    description="First step: Generate multiple quantum circuits from stabilizer generator. This tool uses reinforcement learning(RL) and multiple Qiskit classical algorithms to create candidate circuits, and return in JSON form for further analysis.",
)
async def generate_circuits(
    stabilizers: Annotated[List[str], "Define the stabilizer generator list of quantum codes, such as 7-bit GHZ state as ['+ZZ_____', '+_ZZ____', '+XXXXXXX']."],
    num_rl_circuits: Annotated[int, "The number of RL circuit variants to be generated."] = 3,
) -> list[TextContent]:
    """
    Generate multiple quantum circuits from stabilizer generator, including RL optimized results and Qiskit baseline results.
    """
    if not stabilizers or not isinstance(stabilizers, list):
        return [TextContent(type="text", text="Error: Stabilizer must be a non-empty string list.")]
    
    stabilizers = [s.strip().lstrip('+') for s in stabilizers]
    
    try:
        clean_stabilizers = [s.replace('_', 'I') for s in stabilizers]
        num_qubits = len(clean_stabilizers[0])
        target_tableau = stim.Tableau.from_stabilizers(
            [stim.PauliString(s) for s in clean_stabilizers], allow_underconstrained=True
        )
    except Exception as e:
        return [TextContent(type="text", text=f"Error: Error occurs in analyzing stabilizers: {e}")]

    try:
        project_root = resources.files('gate_optimize').parent.parent
    except (ImportError, AttributeError):
        from pathlib import Path
        project_root = Path(__file__).resolve().parent.parent.parent.parent
    
    params_path = str(project_root / 'model' / 'eval' / '7bit_params.json')
    args = parse(['-fromjson', params_path])
    args.qbits = num_qubits
    utils.set_seed(args.seed)
    
    utils._globals = {
        'debug': False, 'dist': args.dist, 'rewardtype': args.rewardtype, 'swanlab': False,
        'device': utils.get_device(prefer_cuda=True), 'noise': lambda state: state,
        'bufsize': args.bufsize, 'gamma': args.gamma, 'tau': args.tau, 'num_envs': 1
    }
    utils.args = args
    
    exp = Experiment(args.a, training_req=False, n_workers=1)
    
    model_dir = project_root / 'model' / 'plots' / f"{args.qbits}-{args.tol}-{args.name}--{args.exptdate}"
    model_path = str(model_dir / 'model')
    model_exists = os.path.exists(model_path + '.pkl')

    if not model_exists:
        return [TextContent(type="text", text=f"Error: No pre-trained RL model found in {model_path}.pkl.")]

    target_state = stim.Tableau(args.qbits)
    env = exp.initialize_test_env(target_state, target_tableau, args.tol, args.maxsteps, args.gateset, args.dist)
    env.max_steps = int(1 * args.maxsteps)
    
    if not hasattr(exp, 'agent'):
        if args.a in ['ppo', 'vpg']:
            exp.initialize_agent_pg(
                policy_hidden=args.phidden, policy_activ_fn=getattr(torch.nn.functional, args.activfn),
                policy_model_max_grad_norm=0.5, policy_optimizer_fn=lambda net: torch.optim.Adam(net.parameters(), lr=args.plr),
                value_hidden=args.vhidden, value_activ_fn=getattr(torch.nn.functional, args.activfn),
                value_model_max_grad_norm=0.5, value_optimizer_fn=lambda net: torch.optim.Adam(net.parameters(), lr=args.vlr),
                entropy_loss_weight=args.entropywt, gamma=args.gamma,
            )
        exp.load_model(model_path)

    all_circuits_data = []
    
    # Generate RL Circuits
    best_circuits, _ = exp.evaluate(env, n_eps=num_rl_circuits, num_best=num_rl_circuits, verbose=0)
    for i, (actions, _, _, _) in enumerate(best_circuits):
        int_actions = [a.item() if hasattr(a, 'item') else int(a) for a in actions]
        try:
            qc = env.get_inverted_ckt(int_actions)
        except (IndexError, AttributeError):
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
        all_circuits_data.append({"name": f"RL Variant {i+1}", "qasm": qasm3.dumps(qc), "gate_count": len(qc.data)})

    # Generate Qiskit Benchmark Circuits
    qiskit_stabilizers = [s[::-1] for s in clean_stabilizers]
    cliff = qi.StabilizerState.from_stabilizer_list(qiskit_stabilizers, allow_underconstrained=True).clifford
    benchmarks = {
        'bravyi': cliff.to_circuit(),
        'ag': qs.synth_clifford_ag(cliff),
        'bm': qs.synth_clifford_full(cliff),
        'greedy': qs.synth_clifford_greedy(cliff)
    }
    for name, qc in benchmarks.items():
        all_circuits_data.append({"name": f"Qiskit-{name.upper()}", "qasm": qasm3.dumps(qc), "gate_count": len(qc.data)})

    # Finalize output
    output_data = {"num_qubits": num_qubits, "stabilizers": stabilizers, "circuits": all_circuits_data}
    output_json = json.dumps(output_data, indent=2)
    
    summary_text = (f"Successfully generated {len(best_circuits)} RL circuits and {len(benchmarks)} Qiskit baseline circuits.\n"
                    f"Please use the following JSON output for the next tools.")
    return [TextContent(type="text", text=summary_text), TextContent(type="text", text=output_json)]


@mcp.tool(
    name="compare_circuits_fidelity",
    description="Final step: Compares the performance of multiple circuits. It can use either a generic physical noise model or a more accurate one based on calibrated fidelity data if provided.",
)
async def compare_circuits_fidelity(
    circuits_json: Annotated[str, "A JSON string containing the list of circuits generated by the 'generate_circuits' tool."],
    calibrated_fidelities_json: Annotated[str, "Optional: A JSON string with calibrated fidelities from 'analyze_gate_fidelity_from_data'. If not provided, a default physical model is used."] = None,
) -> list[ImageContent | TextContent]:
    """
    Accepts quantum circuits and evaluates their fidelity, using either a default physical noise model or a calibrated one.
    """
    try:
        data = json.loads(circuits_json)
        circuits_info, num_qubits = data["circuits"], data["num_qubits"]
    except (json.JSONDecodeError, KeyError) as e:
        return [TextContent(type="text", text=f"Error: Invalid or incorrect circuits JSON: {e}")]

    results_text_lines, images_base64, fidelity_comparison_data = [], [], []
    
    use_calibrated_model = calibrated_fidelities_json is not None
    title_prefix = "Calibrated" if use_calibrated_model else "Physical Model"
    
    custom_errors = None
    if use_calibrated_model:
        try:
            fidelities_data = json.loads(calibrated_fidelities_json)
            custom_errors = {
                'single_qubit': 1.0 - fidelities_data['single_qubit_fidelity'],
                'two_qubit': 1.0 - fidelities_data['two_qubit_fidelity'],
                'measurement': fidelities_data['spam_error']
            }
            results_text_lines.append(f"\n--- Fidelity Comparison Using Calibrated Error Rates ---")
            results_text_lines.append(f"Using Error Rates: 1Q Gate={custom_errors['single_qubit']:.2e}, 2Q Gate={custom_errors['two_qubit']:.2e}, Readout={custom_errors['measurement']:.2e}")
        except (json.JSONDecodeError, KeyError) as e:
            return [TextContent(type="text", text=f"Error: Invalid calibrated fidelities JSON: {e}")]
    else:
        results_text_lines.append("\n--- Fidelity Comparison Using Generic Physical Error Model ---")

    for circuit_info in circuits_info:
        name, qasm_str, gate_count = circuit_info["name"], circuit_info["qasm"], circuit_info["gate_count"]
        qc = qasm3.loads(qasm_str)
        
        try:
            if use_calibrated_model:
                result = simulate_circuit_fidelity_custom(qc, custom_errors=custom_errors, seed=42)
            else:
                qc_no_meas = qc.remove_final_measurements(inplace=False) if any(i.operation.name == 'measure' for i in qc.data) else qc
                result = simulate_circuit_fidelity(qc_no_meas, error_model='physical', num_shots=1000, seed=42)
            
            fidelity_comparison_data.append({'method': name, 'gate_count': gate_count, 'physical_fidelity': result['fidelity'], 'physical_error_rate': result['error_rate']})
            
            plt.close('all')
            fig, ax = plt.subplots(figsize=(14, max(4, num_qubits * 0.5)))
            qc.draw('mpl', ax=ax, fold=-1)
            ax.set_title(f"{name} | Gates: {gate_count} | {title_prefix} Fidelity: {result['fidelity']:.5f}", fontsize=12, pad=20)
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
            img_buffer.seek(0)
            images_base64.append(base64.b64encode(img_buffer.getvalue()).decode("utf-8"))
            plt.close(fig)
        except Exception as e:
            results_text_lines.append(f"Warning: Failed to process circuit '{name}': {e}")
            fidelity_comparison_data.append({'method': name, 'gate_count': gate_count, 'physical_fidelity': None, 'physical_error_rate': None})


    header = f"{'Method':<18} {'Gates':<6} {'Physical Fidelity':<18} {'Error Rate':<12}"
    results_text_lines.append(header)
    results_text_lines.append("-" * len(header))
    sorted_comparison = sorted(fidelity_comparison_data, key=lambda x: x.get('physical_fidelity') or 0.0, reverse=True)
    for entry in sorted_comparison:
        phys_fid_str = f"{entry['physical_fidelity']:.6f}" if entry['physical_fidelity'] is not None else "N/A"
        error_rate_str = f"{entry['physical_error_rate']:.5f}" if entry['physical_error_rate'] is not None else "N/A"
        results_text_lines.append(f"{entry['method']:<18} {entry['gate_count']:<6} {phys_fid_str:<18} {error_rate_str:<12}")

    final_text = "\n".join(results_text_lines)
    send_to_gui(status="finished", tool_name="compare_circuits_fidelity", text_result=final_text, main_images=images_base64)
    return [TextContent(type="text", text=final_text)] + [ImageContent(type="image", data=b64, mimeType="image/png") for b64 in images_base64]

@mcp.tool(
    name="plot_timeline",
    description="Plot execution timeline for quantum circuits. Takes a JSON object containing multiple circuits and generates timeline visualization showing when each qubit is busy with operations.",
)
async def plot_timeline(
    circuits_json: Annotated[str, "A JSON string, containing circuit list generated by 'generate_circuits' tool."],
    circuit_index: Annotated[int, "Index of the circuit to plot timeline for (0-based)."] = 0,
    title_suffix: Annotated[str, "Additional text to add to the plot title."] = "",
) -> list[ImageContent | TextContent]:
    """
    Plot execution timeline for a quantum circuit showing when each qubit is busy with operations.
    """
    try:
        data = json.loads(circuits_json)
        circuits_info = data["circuits"]
    except (json.JSONDecodeError, KeyError) as e:
        return [TextContent(type="text", text=f"Error: Invalid or incorrect input JSON: {e}")]

    if not circuits_info:
        return [TextContent(type="text", text="Error: Empty circuit list. Unable to plot timeline.")]

    if circuit_index < 0 or circuit_index >= len(circuits_info):
        return [TextContent(type="text", text=f"Error: Circuit index {circuit_index} out of range. Available circuits: 0-{len(circuits_info)-1}.")]

    # Get the selected circuit
    circuit_info = circuits_info[circuit_index]
    circuit_name = circuit_info['name']
    qasm_str = circuit_info['qasm']
    gate_count = circuit_info['gate_count']
    
    try:
        # Parse QASM to Qiskit circuit
        qc = qasm3.loads(qasm_str)
        
        # Generate timeline plot using the utils function
        timeline_img_base64 = utils.plot_timeline_qiskit(
            qc, 
            save_path=None,  # Return base64 instead of saving
            title_suffix=f" - {circuit_name}{title_suffix}"
        )
        
        summary_text = [
            f"--- Timeline Plot for Circuit '{circuit_name}' ---",
            f"Circuit: {circuit_name}",
            f"Total gates: {gate_count}",
            f"Number of qubits: {qc.num_qubits}",
            f"Timeline visualization shows the execution schedule including:",
            f"  - Single qubit operations (blue)",
            f"  - Two qubit operations (red)", 
            f"  - Inter-zone movement (yellow)",
            f"  - Intra-zone movement (green)",
            f"  - Idle time (gray)",
            f"The plot helps visualize parallelization opportunities and bottlenecks."
        ]
        
        final_text = "\n".join(summary_text)
        
        # Send to GUI
        send_to_gui(
            status="finished", 
            tool_name="plot_timeline",
            text_result=final_text, 
            main_images=[timeline_img_base64]
        )
        
        return [
            TextContent(type="text", text=final_text),
            ImageContent(type="image", data=timeline_img_base64, mimeType="image/png")
        ]
        
    except Exception as e:
        return [TextContent(type="text", text=f"Error: Failed to plot timeline for circuit '{circuit_name}': {e}")]


@mcp.tool(
    name="simplify_best_circuit",
    description="Third step: Simplify the best quantum circuits. Accepts a JSON object containing multiple circuits, choosing the circuit with least gate number. Use Qiskit transpile function for depth optimization and simplification.",
)
async def simplify_best_circuit(
    circuits_json: Annotated[str, "A JSON string, containing circuit list generated by 'generate_circuits' tool."],
) -> list[ImageContent | TextContent]:
    """
    Choose the best circuit (according to gate number) from a series of circuits, and simplify.
    """
    try:
        data = json.loads(circuits_json)
        circuits_info = data["circuits"]
    except (json.JSONDecodeError, KeyError) as e:
        return [TextContent(type="text", text=f"Error: Invalid or incorrect input JSON: {e}")]

    if not circuits_info:
        return [TextContent(type="text", text="Error: Empty circuit list. Unable to simplify.")]

    best_circuit_info = min(circuits_info, key=lambda x: x['gate_count'])
    best_qc_name = best_circuit_info['name']
    best_qc_qasm = best_circuit_info['qasm']
    best_qc = qasm3.loads(best_qc_qasm)
    
    summary = [f"--- Circuit simplification ---",
               f"Best initial circuit chosen for simplification: '{best_qc_name}' (Gate number: {len(best_qc.data)})."]

    simplified_qc = transpile(best_qc, basis_gates=['h', 's', 'cx', 'sdg'], optimization_level=3)
    summary.append(f"After simplification, gate number is reduced to: {len(simplified_qc.data)}.")
    summary.append("Comparison summary before/after simplification generated.")

    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
    
    best_qc.draw('mpl', ax=ax1, fold=-1)
    ax1.set_title(f"Before simplification: '{best_qc_name}' (Gate number: {len(best_qc.data)})", fontsize=14)
    
    simplified_qc.draw('mpl', ax=ax2, fold=-1)
    ax2.set_title(f"After simplification (Gate number: {len(simplified_qc.data)})", fontsize=14)
    
    plt.tight_layout(pad=3.0)
    
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    simplification_img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    plt.close(fig)

    final_text = "\n".join(summary)
    send_to_gui(
        status="finished", tool_name="simplify_best_circuit",
        text_result=final_text, main_images=[simplification_img_base64]
    )
    
    return [
        TextContent(type="text", text=final_text),
        ImageContent(type="image", data=simplification_img_base64, mimeType="image/png")
    ]

@mcp.tool(
    name="simulate_gate_benchmark_data",
    description="Calibration Step 1: Generates a standardized set of simulated experimental data for gate fidelity benchmarking on a neutral atom platform. It creates data for both single-qubit (X) and two-qubit (CZ) gates using fixed, realistic parameters.",
)
async def simulate_gate_benchmark_data() -> list[TextContent]:
    """
    Generates a standardized, simulated dataset for gate fidelity calibration.
    This tool uses a fixed set of internal parameters and requires no input.
    """
    params = {
        'X': {'true_fidelity': 0.999, 'num_states': 2},
        'CZ': {'true_fidelity': 0.995, 'num_states': 4},
        'shared': {'true_spam_error': 0.02, 'num_shots': 8192, 'max_sequence_length': 100, 'num_points': 20}
    }
    output_data = {}
    for gate_type in ['X', 'CZ']:
        gate_params = params[gate_type]
        shared_params = params['shared']
        true_fidelity, num_states = gate_params['true_fidelity'], gate_params['num_states']
        true_spam_error, num_shots, max_len, num_pts = shared_params.values()
        m_values = np.unique(np.logspace(0, np.log10(max_len), num_pts, dtype=int))
        if gate_type == 'X':
            m_values = m_values[m_values % 2 != 0]
        B, A = 1.0 / num_states, 1.0 - true_spam_error - (1.0 / num_states)
        def ideal_survival_prob(m, p): return A * (p ** m) + B
        measured_probs, error_bars = [], []
        for m in m_values:
            prob = ideal_survival_prob(m, true_fidelity)
            successes = np.random.binomial(num_shots, np.clip(prob, 0, 1))
            measured_p = successes / num_shots
            measured_probs.append(measured_p)
            error_bars.append(np.sqrt(measured_p * (1 - measured_p) / num_shots))
        output_data[f"{gate_type}_gate_data"] = {"sequence_lengths": m_values.tolist(), "measured_probabilities": measured_probs, "error_bars": error_bars}
    output_data["metadata"] = params['shared']
    summary = f"Generated benchmark data for both X and CZ gates with {params['shared']['num_shots']} shots. The JSON output can now be passed to `analyze_gate_fidelity_from_data`."
    return [TextContent(type="text", text=summary), TextContent(type="text", text=json.dumps(output_data, indent=2))]

@mcp.tool(
    name="analyze_gate_fidelity_from_data",
    description="Calibration Step 2: Analyzes experimental data to extract gate fidelities and SPAM error. It fits the data for single-qubit (X) and two-qubit (CZ) gates, visualizes the results, and returns the calibrated parameters.",
)
async def analyze_gate_fidelity_from_data(
    benchmark_data_json: Annotated[str, "A JSON string containing benchmark data for both X and CZ gates, generated by `simulate_gate_benchmark_data`."],
) -> list[ImageContent | TextContent]:
    """
    Fits experimental data for X and CZ gates to extract their fidelities and the SPAM error.
    """
    try:
        data = json.loads(benchmark_data_json)
        x_data, cz_data = data["X_gate_data"], data["CZ_gate_data"]
    except (json.JSONDecodeError, KeyError) as e:
        return [TextContent(type="text", text=f"Error: Invalid or incomplete input JSON: {e}")]
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    calibrated_results = {}
    def decay_func(m, A, p, B): return A * (p ** m) + B
    for gate_type, gate_data, ax in [('X', x_data, ax1), ('CZ', cz_data, ax2)]:
        m_values, probs, errors = np.array(gate_data["sequence_lengths"]), np.array(gate_data["measured_probabilities"]), np.array(gate_data["error_bars"])
        num_states = 2 if gate_type == 'X' else 4
        p0, bounds = [1.0 - 1.0/num_states, 0.99, 1.0/num_states], ([0, 0, 0], [1, 1.0001, 1])
        try:
            params, _ = curve_fit(decay_func, m_values, probs, p0=p0, bounds=bounds, sigma=errors, absolute_sigma=True)
            A_fit, p_fit, B_fit = params
            fidelity, spam_error = np.clip(p_fit, 0, 1), 1.0 - (A_fit + B_fit)
            calibrated_results[gate_type] = {'fidelity': fidelity, 'spam_error': spam_error}
            ax.errorbar(m_values, probs, yerr=errors, fmt='o', color='blue', label='Simulated Data w/ Error Bars', zorder=5)
            m_fine = np.linspace(0, max(m_values), 200)
            ax.plot(m_fine, decay_func(m_fine, *params), label='Exponential Decay Fit', color='red', linewidth=2)
            results_text = f"Fit Results:\nGate Fidelity (p) = {fidelity:.5f}\nSPAM Error = {spam_error:.4f}"
            ax.text(0.55, 0.65, results_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        except RuntimeError:
            ax.text(0.5, 0.5, "Fitting Failed", ha='center', va='center', fontsize=16, color='red')
        ax.set_ylabel("Survival Probability", fontsize=12)
        ax.set_title(f"{gate_type} Gate Fidelity Analysis", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
    ax2.set_xlabel("Sequence Length (Number of Repeated Gates, m)", fontsize=12)
    fig.tight_layout(pad=3.0)
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    plt.close(fig)
    avg_spam = np.mean([res['spam_error'] for res in calibrated_results.values()])
    output_data = {"single_qubit_fidelity": calibrated_results.get('X', {}).get('fidelity'), "two_qubit_fidelity": calibrated_results.get('CZ', {}).get('fidelity'), "spam_error": avg_spam}
    summary = (f"Successfully analyzed benchmark data.\n" f"Calibrated 1Q Fidelity: {output_data['single_qubit_fidelity']:.5f}, " f"Calibrated 2Q Fidelity: {output_data['two_qubit_fidelity']:.5f}, " f"Average SPAM Error: {avg_spam:.4f}\n" "This JSON output is ready for the final evaluation step.")
    return [ImageContent(type="image", data=img_base64, mimeType="image/png"), TextContent(type="text", text=summary), TextContent(type="text", text=json.dumps(output_data, indent=2))]

# Import and register QEC tools
from . import server_qec

@mcp.tool(
    name="analyze_qec_logical_error_rate",
    description="Analyze quantum error correction codes by calculating logical error rates vs physical error rates using MWPM and BP-OSD decoders.",
)
async def analyze_qec_logical_error_rate(
    stabilizers: Annotated[List[str], "List of stabilizer strings for the quantum error correction code (e.g., ['+ZZ_____', '+_ZZ____', '+XXXXXXX'] for 7-qubit Steane code)"],
    logical_Z_operators: Annotated[List[str], "List of logical Z operator strings for the code (e.g., ['ZZZZZZZ'] for Steane code logical X and Z operators). Optional for auto-detection."] = None,
    rounds: Annotated[int, "Number of syndrome measurement rounds"] = 3,
    decoder_method: Annotated[str, "Decoding method: 'mwpm', 'bp_osd', or 'both' for comparison"] = 'mwpm',
) -> list[ImageContent | TextContent]:
    return await server_qec.analyze_qec_logical_error_rate(
        stabilizers, logical_Z_operators, rounds, decoder_method
    )


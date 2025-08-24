# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm
from mpl_toolkits.mplot3d import Axes3D # For Bloch sphere
import os
# %matplotlib qt
# --- Physical constants and Pauli matrices ---
sigmax = np.array([[0, 1], [1, 0]], dtype=complex)
sigmay = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigmaz = np.array([[1, 0], [0, -1]], dtype=complex)
I_2 = np.eye(2, dtype=complex)
H1_op = sigmax / 2
H2_op = sigmay / 2
H_detuning_op = sigmaz # For applying detuning explicitly

def load_fourier_coeffs_from_csv(filename):
    """Loads Fourier coefficients from a CSV file."""
    if not os.path.exists(filename):
        # print(f"Error: File '{filename}' not found.")
        return None, None
    try:
        df = pd.read_csv(filename, index_col='n') 
        a_coeffs_loaded = None
        b_coeffs_loaded = None
        if 'a_coeffs' in df.columns:
            a_coeffs_loaded = df['a_coeffs'].values
        if 'b_coeffs' in df.columns:
            b_coeffs_loaded = df['b_coeffs'].values
        
        if a_coeffs_loaded is None:
            # print("Error: Column 'a_coeffs' not found in CSV file.")
            return None, None
        if b_coeffs_loaded is None:
            # print("Warning: Column 'b_coeffs' not found in CSV file. Assuming b_coeffs are zero.")
            b_coeffs_loaded = np.zeros_like(a_coeffs_loaded)

        if len(a_coeffs_loaded) < 1 or len(b_coeffs_loaded) < 1 :
            #  print("Error: Coefficient array length is insufficient.")
             return None,None
        
        if len(a_coeffs_loaded) != len(b_coeffs_loaded) and np.any(b_coeffs_loaded):
            # print(f"Warning: Length mismatch between a_coeffs (len {len(a_coeffs_loaded)}) and b_coeffs (len {len(b_coeffs_loaded)}). Truncating/padding b_coeffs to match a_coeffs.")
            b_new = np.zeros_like(a_coeffs_loaded)
            min_len = min(len(a_coeffs_loaded), len(b_coeffs_loaded))
            b_new[:min_len] = b_coeffs_loaded[:min_len]
            b_coeffs_loaded = b_new

        # print(f"Successfully loaded Fourier coefficients from '{filename}'.")
        return a_coeffs_loaded, b_coeffs_loaded
    except Exception as e:
        # print(f"Error reading coefficients CSV file: {e}")
        return None, None

def reconstruct_phi_array(t_points_array, L_pulse, a_coeffs, b_coeffs):
    """Reconstructs phase phi(t) at given time points using Fourier coefficients."""
    phi_reconstructed = np.zeros_like(t_points_array, dtype=float)
    if len(a_coeffs) > 0:
        phi_reconstructed += a_coeffs[0] / 2.0  
    omega_n_factor = 2.0 * np.pi / L_pulse
    for n in range(1, len(a_coeffs)):
        current_omega_n = n * omega_n_factor
        phi_reconstructed += a_coeffs[n] * np.cos(current_omega_n * t_points_array)
    if b_coeffs is not None and len(b_coeffs) > 1:
        for n in range(1, len(b_coeffs)):
            current_omega_n = n * omega_n_factor
            phi_reconstructed += b_coeffs[n] * np.sin(current_omega_n * t_points_array)
    return np.mod(phi_reconstructed, 2 * np.pi)

def get_hamiltonian_at_t(phi_val, current_omega_max, current_detuning_val):
    """Constructs the Hamiltonian for given phi, omega_max, and detuning."""
    H_control = current_omega_max * (np.cos(phi_val) * H1_op - np.sin(phi_val) * H2_op)
    H_detune = current_detuning_val * H_detuning_op # H_detuning_op is sigma_z
    return H_control + H_detune

def state_to_bloch_vector(psi):
    """Converts quantum state |psi> = [alpha, beta]^T to Bloch sphere coordinates."""
    psi = psi / np.linalg.norm(psi)
    sx = (psi.conj().T @ sigmax @ psi)[0,0]
    sy = (psi.conj().T @ sigmay @ psi)[0,0]
    sz = (psi.conj().T @ sigmaz @ psi)[0,0]
    return np.real(sx), np.real(sy), np.real(sz)

def simulate_evolution(initial_psi, t_array, dt_sim, L_pulse_sim, 
                       a_c, b_c, omega_max_sim_val, detuning_sim_val):
    """Simulates quantum evolution for a given set of parameters."""
    psi_current_sim = initial_psi.copy()
    bloch_history_sim = [state_to_bloch_vector(psi_current_sim)]
    num_steps_sim = len(t_array)

    for k_s in range(num_steps_sim):
        t_mid_s = t_array[k_s] + dt_sim / 2.0
        phi_at_t_mid_s = reconstruct_phi_array(np.array([t_mid_s]), L_pulse_sim, a_c, b_c)[0]
        H_s = get_hamiltonian_at_t(phi_at_t_mid_s, omega_max_sim_val, detuning_sim_val)
        U_s = expm(-1j * H_s * dt_sim)
        psi_current_sim = U_s @ psi_current_sim
        bloch_history_sim.append(state_to_bloch_vector(psi_current_sim))
    return psi_current_sim, bloch_history_sim

def plot_bloch_sphere_comparison(bloch_hist_ideal, error_cases_data, fig_title_prefix=""):
    """
    Plots 4 subplots on a Bloch sphere, each comparing ideal trajectory with one error case.
    error_cases_data: list of tuples, each tuple is (label_str, bloch_hist_error)
    """
    if len(error_cases_data) != 4:
        # print("Error: plot_bloch_sphere_comparison expects exactly 4 error cases.")
        return

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(f"{fig_title_prefix}Bloch Sphere Trajectory Comparisons", fontsize=16)

    for i, (label, bloch_hist_error) in enumerate(error_cases_data):
        ax = fig.add_subplot(2, 2, i + 1, projection='3d')

        # Draw Bloch sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x_sphere, y_sphere, z_sphere, color='c', alpha=0.05, linewidth=0)

        # Plot ideal trajectory
        bx_ideal, by_ideal, bz_ideal = zip(*bloch_hist_ideal)
        ax.plot(bx_ideal, by_ideal, bz_ideal, color='blue', linestyle='-', label='Ideal Trajectory', linewidth=1.5)
        ax.scatter(bx_ideal[0], by_ideal[0], bz_ideal[0], color='blue', s=50, marker='o') # Ideal start
        ax.scatter(bx_ideal[-1], by_ideal[-1], bz_ideal[-1], color='blue', s=80, marker='X') # Ideal end


        # Plot error trajectory
        bx_err, by_err, bz_err = zip(*bloch_hist_error)
        ax.plot(bx_err, by_err, bz_err, color='red', linestyle='--', label=f'{label} Trajectory', linewidth=1.5)
        # Start point of error trajectory is the same as ideal
        ax.scatter(bx_err[-1], by_err[-1], bz_err[-1], color='red', s=80, marker='P') # Error end ('P' for Plus)

        ax.set_xlabel('⟨σx⟩')
        ax.set_ylabel('⟨σy⟩')
        ax.set_zlabel('⟨σz⟩')
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_zlim([-1.1, 1.1])
        ax.set_title(f"Ideal vs. {label}")
        ax.legend(fontsize='small')
        ax.view_init(elev=20., azim=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plt.show()


if __name__ == "__main__":
    coeffs_csv_filename = "optimized_fourier_coeffs_20250612-1804.csv" 
    total_pulse_duration_L = np.pi * 2.5
    num_evolution_steps = 50   
    omega_max_ideal = 1.0      
    
    a_coeffs, b_coeffs_loaded = load_fourier_coeffs_from_csv(coeffs_csv_filename)

    if a_coeffs is not None:
        psi_initial_state = np.array([[1], [0]], dtype=complex) # |0>

        time_array_evol = np.linspace(0, total_pulse_duration_L, num_evolution_steps, endpoint=False)
        dt_evol = total_pulse_duration_L / num_evolution_steps
        
        # print("\nSimulating Ideal Case...")
        final_psi_ideal, bloch_hist_ideal = simulate_evolution(
            psi_initial_state, time_array_evol, dt_evol, total_pulse_duration_L,
            a_coeffs, b_coeffs_loaded, omega_max_ideal, 0.0 # Ideal: omega_max_ideal, detuning=0
        )
        phase_ideal = np.angle(final_psi_ideal[1,0])
        fid_ideal = np.abs(final_psi_ideal[1,0])**2
        # print(f"Ideal - Final state pop in |1>: {fid_ideal:.6f}, Phase of |1>: {phase_ideal:.4f} rad")

        # Define error cases
        error_factor = 0.05
        error_cases_params = [
            {"label": "Ω_max +5%", "omega_factor": 1 + error_factor, "detuning_val": 0.0},
            {"label": "Ω_max -5%", "omega_factor": 1 - error_factor, "detuning_val": 0.0},
            {"label": "Δ = +5% Ω_0", "omega_factor": 1.0, "detuning_val": error_factor * omega_max_ideal},
            {"label": "Δ = -5% Ω_0", "omega_factor": 1.0, "detuning_val": -error_factor * omega_max_ideal},
        ]
        
        error_case_results_for_plot = []

        for case_params in error_cases_params:
            # print(f"\nSimulating Case: {case_params['label']}...")
            current_omega = omega_max_ideal * case_params['omega_factor']
            current_detuning = case_params['detuning_val']
            
            final_psi_error, bloch_hist_error = simulate_evolution(
                psi_initial_state, time_array_evol, dt_evol, total_pulse_duration_L,
                a_coeffs, b_coeffs_loaded, current_omega, current_detuning
            )
            phase_error = np.angle(final_psi_error[1,0]) if np.abs(final_psi_error[1,0]) > 1e-3 else np.nan
            fid_error = np.abs(final_psi_error[1,0])**2
            # print(f"{case_params['label']} - Final state pop in |1>: {fid_error:.6f}, Phase of |1>: {phase_error:.4f} rad")
            error_case_results_for_plot.append((case_params['label'], bloch_hist_error))
            
        plot_bloch_sphere_comparison(bloch_hist_ideal, error_case_results_for_plot)
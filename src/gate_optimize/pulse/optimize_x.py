# %%
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import expm
import itertools
import pandas as pd
from datetime import datetime
import os

# %matplotlib qt

# optmize X gate
class RobustGRAPEFourierCoeffs():
    def __init__(self, t_final=np.pi, N_slices_for_evolution=50, 
                 num_fourier_terms=10, omega_max_val=1.0, 
                 rise_time=0.0, fall_time=0.0, # New parameters
                 initial_coeffs_file=None, 
                 optimize_sine_terms=True):

        self.sigmax = np.array([[0, 1], [1, 0]], dtype=complex)
        self.sigmay = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.sigmaz = np.array([[1, 0], [0, -1]], dtype=complex)
        self.I = np.eye(2, dtype=complex)
        self.H1 = self.sigmax / 2 
        self.H2 = self.sigmay / 2
        self.H_detuning_term = self.sigmaz 

        self.initial_state = np.array([[1], [0]], dtype=complex)
        self.target_state = np.array([[0], [1]], dtype=complex)
        
        self.Omega_max = omega_max_val 
        self.t_final = t_final 
        self.N_slices_for_evolution = N_slices_for_evolution 
        self.dt_evolution = self.t_final / self.N_slices_for_evolution
        self.num_fourier_terms = num_fourier_terms
        self.optimize_sine_terms = optimize_sine_terms
        
        self.rise_time = rise_time
        self.fall_time = fall_time
        self.omega_envelope = self._create_amplitude_envelope()

        
        self.num_coeffs = 1 + (2 if self.optimize_sine_terms else 1) * self.num_fourier_terms
        self.a_coeffs = np.zeros(self.num_fourier_terms + 1) 
        self.b_coeffs = np.zeros(self.num_fourier_terms + 1) 

        loaded_successfully = False
        if initial_coeffs_file and os.path.exists(initial_coeffs_file):
            try:
                df_coeffs = pd.read_csv(initial_coeffs_file)
                can_load_a = 'a_coeffs' in df_coeffs.columns and len(df_coeffs['a_coeffs'].values) == self.num_fourier_terms + 1
                can_load_b = 'b_coeffs' in df_coeffs.columns and len(df_coeffs['b_coeffs'].values) == self.num_fourier_terms + 1
                
                if can_load_a:
                    self.a_coeffs = df_coeffs['a_coeffs'].values
                    print(f"Successfully loaded initial a_coeffs from {initial_coeffs_file}")
                    loaded_successfully = True
                else:
                     print(f"Warning: Could not load a_coeffs or length mismatch from {initial_coeffs_file}.")

                if self.optimize_sine_terms and can_load_b:
                    self.b_coeffs = df_coeffs['b_coeffs'].values
                    print(f"Successfully loaded initial b_coeffs from {initial_coeffs_file}")
                elif self.optimize_sine_terms and not can_load_b:
                    print(f"Warning: Sine term optimization enabled, but could not load b_coeffs or length mismatch from {initial_coeffs_file}. b_coeffs will be random/zero.")
                    if not loaded_successfully:
                        self.a_coeffs[1:] = np.random.randn(self.num_fourier_terms) * 0.1
                    self.b_coeffs[1:] = np.random.randn(self.num_fourier_terms) * 0.1
                elif not self.optimize_sine_terms:
                    print("Sine term optimization is disabled. b_coeffs will remain zero.")
                    self.b_coeffs.fill(0)

            except Exception as e:
                print(f"Error loading initial coefficients from {initial_coeffs_file}: {e}.")
                loaded_successfully = False
        
        if not loaded_successfully:
            # A good guess for a0 for an X gate is around pi, but we let it optimize.
            # Initializing around this value can speed up convergence.
            self.a_coeffs[0] = np.random.uniform(0.5 * np.pi, 1.5 * np.pi) * 2 
            self.a_coeffs[1:] = np.random.randn(self.num_fourier_terms) * 0.1
            if self.optimize_sine_terms:
                self.b_coeffs[1:] = np.random.randn(self.num_fourier_terms) * 0.1 
            else:
                self.b_coeffs.fill(0)
            print("Initialized Fourier coefficients (a0 for DC offset, others small random).")
        
        
        self.fidelity = None
        self.rabi_factors_opt = np.array([0.98, 1.0, 1.02])
        self.detuning_multipliers_opt = np.array([-0.02,  0.0, 0.02])
        
        self.noise_conditions = [] 
        for r_factor in self.rabi_factors_opt:
            for d_multiplier in self.detuning_multipliers_opt:
                self.noise_conditions.append({
                    'name': f'R{r_factor:.2f}_D{d_multiplier:.2f}',
                    'omega_factor': r_factor,
                    'detuning_val': d_multiplier * self.Omega_max
                })
        self.opt_grid_rabi_factors = self.rabi_factors_opt
        self.opt_grid_detuning_multipliers = self.detuning_multipliers_opt
        self.individual_fidelities_history = {cond['name']: [] for cond in self.noise_conditions}
        
        self.adam_t = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m_a_coeffs = np.zeros_like(self.a_coeffs)
        self.v_a_coeffs = np.zeros_like(self.a_coeffs)
        self.m_b_coeffs = np.zeros_like(self.b_coeffs)
        self.v_b_coeffs = np.zeros_like(self.b_coeffs)

    def _create_amplitude_envelope(self):
        if self.rise_time + self.fall_time > self.t_final:
            print(f"Warning: rise_time ({self.rise_time}) + fall_time ({self.fall_time}) "
                  f"> t_final ({self.t_final}). The ramps will overlap.")

        t_centers = (np.arange(self.N_slices_for_evolution) + 0.5) * self.dt_evolution
        omega_values = np.zeros(self.N_slices_for_evolution)

        for i, t in enumerate(t_centers):
            if t < self.rise_time:
                omega_values[i] = self.Omega_max * (t / self.rise_time) if self.rise_time > 0 else self.Omega_max
            elif t > self.t_final - self.fall_time:
                time_into_fall = t - (self.t_final - self.fall_time)
                omega_values[i] = self.Omega_max * (1 - time_into_fall / self.fall_time) if self.fall_time > 0 else self.Omega_max
            else:
                omega_values[i] = self.Omega_max
        
        return np.maximum(0, omega_values)

    def reconstruct_phi_at_t(self, t_points, a_coeffs_curr, b_coeffs_curr):
        phi_reconstructed = np.zeros_like(t_points, dtype=float)
        if len(a_coeffs_curr) > 0:
            phi_reconstructed += a_coeffs_curr[0] / 2.0
        
        # Fourier series is defined over interval [-L, L] or [0, L]. Here we use [0, L] convention.
        # So omega_n = n * 2*pi / L.
        omega_n_factor = 2.0 * np.pi / self.t_final
        
        # Using vectorization for speed
        n_vals = np.arange(1, len(a_coeffs_curr)).reshape(-1, 1)
        t_vals = np.atleast_2d(t_points)
        cos_terms = np.cos(n_vals * omega_n_factor * t_vals)
        phi_reconstructed += np.sum(a_coeffs_curr[1:, np.newaxis] * cos_terms, axis=0)

        if self.optimize_sine_terms or np.any(b_coeffs_curr[1:] != 0):
            sin_terms = np.sin(n_vals * omega_n_factor * t_vals)
            phi_reconstructed += np.sum(b_coeffs_curr[1:, np.newaxis] * sin_terms, axis=0)
            
        return np.mod(phi_reconstructed, 2 * np.pi)

    def get_H_k_fourier(self, k_slice_idx_evol, omega_factor, detuning_val, a_c, b_c):
        t_mid = (k_slice_idx_evol + 0.5) * self.dt_evolution
        phi_at_t_mid = self.reconstruct_phi_at_t(np.array([t_mid]), a_c, b_c)[0]
        
        omega_k = self.omega_envelope[k_slice_idx_evol]
        H_control_k = omega_factor * omega_k * \
                      (np.cos(phi_at_t_mid) * self.H1 - np.sin(phi_at_t_mid) * self.H2)

                      
        H_static_noise_k = detuning_val * self.H_detuning_term
        return H_control_k + H_static_noise_k
    
    def get_U_indt_list_fourier(self, omega_factor, detuning_val, a_c, b_c):
        U_list = []
        for k_evol in range(self.N_slices_for_evolution):
            H_k_evol = self.get_H_k_fourier(k_evol, omega_factor, detuning_val, a_c, b_c)
            U_list.append(expm(-1j * H_k_evol * self.dt_evolution))
        return U_list

    def calculate_total_evolution_operator(self, U_indt_list_for_cond):
        U_total = self.I
        for k_idx in range(len(U_indt_list_for_cond)):
             U_total = U_indt_list_for_cond[k_idx] @ U_total 
        return U_total
        
    def calculate_average_fidelity_fourier(self, a_c, b_c, noise_conditions_set, store_individual_fids_to_history=False):
        total_fidelity_sum = 0.0
        current_individual_fids_dict = {} 
        for cond in noise_conditions_set:
            U_list_cond = self.get_U_indt_list_fourier(cond['omega_factor'], cond['detuning_val'], a_c, b_c)
            U_total_cond = self.calculate_total_evolution_operator(U_list_cond)
            psi_final_cond = U_total_cond @ self.initial_state
            C_if_cond = (self.target_state.conj().T @ psi_final_cond)[0,0]
            F_cond = np.abs(C_if_cond)**2
            total_fidelity_sum += F_cond
            current_individual_fids_dict[cond['name']] = F_cond
            if store_individual_fids_to_history and cond['name'] in self.individual_fidelities_history:
                 self.individual_fidelities_history[cond['name']].append(F_cond)
        avg_fid = total_fidelity_sum / len(noise_conditions_set) if noise_conditions_set else 0.0
        return avg_fid, current_individual_fids_dict

    def iteration_onestep_numerical_derivative(self, lr=0.01, d_coeff=1e-6):
        grad_a_coeffs = np.zeros_like(self.a_coeffs)
        grad_b_coeffs = np.zeros_like(self.b_coeffs) 
        current_optimization_conditions = self.noise_conditions
        f_avg_original, _ = self.calculate_average_fidelity_fourier(
            self.a_coeffs, self.b_coeffs, current_optimization_conditions
        )

        for i in range(len(self.a_coeffs)):
            a_coeffs_perturbed = self.a_coeffs.copy()
            a_coeffs_perturbed[i] += d_coeff
            f_avg_perturbed, _ = self.calculate_average_fidelity_fourier(
                a_coeffs_perturbed, self.b_coeffs, current_optimization_conditions
            )
            grad_a_coeffs[i] = (f_avg_perturbed - f_avg_original) / d_coeff
        
        if self.optimize_sine_terms:
            for i in range(1, len(self.b_coeffs)): 
                b_coeffs_perturbed = self.b_coeffs.copy()
                b_coeffs_perturbed[i] += d_coeff
                f_avg_perturbed, _ = self.calculate_average_fidelity_fourier(
                    self.a_coeffs, b_coeffs_perturbed, current_optimization_conditions
                )
                grad_b_coeffs[i] = (f_avg_perturbed - f_avg_original) / d_coeff
            
        self.adam_t += 1
        self.m_a_coeffs = self.beta1 * self.m_a_coeffs + (1 - self.beta1) * grad_a_coeffs
        self.v_a_coeffs = self.beta2 * self.v_a_coeffs + (1 - self.beta2) * (grad_a_coeffs ** 2)
        m_a_hat = self.m_a_coeffs / (1 - self.beta1 ** self.adam_t) if self.adam_t > 0 else self.m_a_coeffs
        v_a_hat = self.v_a_coeffs / (1 - self.beta2 ** self.adam_t) if self.adam_t > 0 else self.v_a_coeffs
        self.a_coeffs += lr * m_a_hat / (np.sqrt(np.abs(v_a_hat)) + self.epsilon)
        
        if self.optimize_sine_terms:
            self.m_b_coeffs[1:] = self.beta1 * self.m_b_coeffs[1:] + (1 - self.beta1) * grad_b_coeffs[1:]
            self.v_b_coeffs[1:] = self.beta2 * self.v_b_coeffs[1:] + (1 - self.beta2) * (grad_b_coeffs[1:] ** 2)
            m_b_hat_update = self.m_b_coeffs / (1 - self.beta1 ** self.adam_t) if self.adam_t > 0 else self.m_b_coeffs
            v_b_hat_update = self.v_b_coeffs / (1 - self.beta2 ** self.adam_t) if self.adam_t > 0 else self.v_b_coeffs
            self.b_coeffs[1:] += lr * m_b_hat_update[1:] / (np.sqrt(np.abs(v_b_hat_update[1:])) + self.epsilon)
        
        self.fidelity, current_fids_dict = self.calculate_average_fidelity_fourier(
            self.a_coeffs, self.b_coeffs, current_optimization_conditions, 
            store_individual_fids_to_history=True
        )
        return self.fidelity, current_fids_dict

    def PWC_pulse(self, pwc_pulse_y_values):
        time_slice_starts = np.linspace(0, self.t_final - self.dt_evolution, self.N_slices_for_evolution)
        t_stair = []
        y_stair = []
        if self.N_slices_for_evolution == 0: return np.array([]), np.array([])
        for i in range(self.N_slices_for_evolution):
            t_stair.extend([time_slice_starts[i], time_slice_starts[i] + self.dt_evolution])
            y_stair.extend([pwc_pulse_y_values[i], pwc_pulse_y_values[i]])
        return np.array(t_stair), np.array(y_stair)

    def export_fourier_coeffs_to_csv(self, base_filename="optimized_fourier_coeffs"):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        filename = f"{base_filename}_{timestamp}.csv"
        df = pd.DataFrame({
            'a_coeffs': self.a_coeffs,
            'b_coeffs': self.b_coeffs 
        })
        df.to_csv(filename, index_label='n')
        print(f"Fourier coefficients exported to {filename}")

    def export_reconstructed_pulse_to_csv(self, base_filename="reconstructed_fourier_pulse"):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M")
        filename = f"{base_filename}_{timestamp}.csv"
        
        time_points = (np.arange(self.N_slices_for_evolution) + 0.5) * self.dt_evolution
        phase_values = self.reconstruct_phi_at_t(time_points, self.a_coeffs, self.b_coeffs)
        amplitude_values = self.omega_envelope 
        
        df = pd.DataFrame({
            'Time': time_points,
            'Phase (radians)': phase_values,
            'Amplitude (units of Omega_max)': amplitude_values / self.Omega_max if self.Omega_max > 0 else amplitude_values
        })
        df.to_csv(filename, index=False)
        print(f"Reconstructed PWC pulse from Fourier coeffs exported to {filename}")

    
    def evaluate_and_plot_final_robustness(self, fig_final, axs_final):
        rabi_factors_eval = np.linspace(0.95, 1.05, 11)
        detuning_multipliers_eval = np.linspace(-0.05, 0.05, 11)
        fidelities_eval_grid = np.zeros((len(detuning_multipliers_eval), len(rabi_factors_eval))) 
        
        eval_conditions_list = [{'name': f'Eval_R{r:.3f}_D{d:.3f}', 'omega_factor': r, 'detuning_val': d * self.Omega_max}
                                for d in detuning_multipliers_eval for r in rabi_factors_eval]
        
        all_eval_fids = {}
        for condition in eval_conditions_list:
            fid_val, _ = self.calculate_average_fidelity_fourier(self.a_coeffs, self.b_coeffs, [condition])
            all_eval_fids[condition['name']] = fid_val
            
        for d_idx, d_multiplier in enumerate(detuning_multipliers_eval):
            for r_idx, r_factor in enumerate(rabi_factors_eval):
                cond_name = f'Eval_R{r_factor:.3f}_D{d_multiplier:.3f}'
                fidelities_eval_grid[d_idx, r_idx] = all_eval_fids.get(cond_name, 0.0)
                
        avg_eval_fidelity = np.mean(fidelities_eval_grid)

        axs_final[0].clear()
        t_fine_plot = np.linspace(0, self.t_final, 400)
        phi_pulse_values = self.reconstruct_phi_at_t(t_fine_plot, self.a_coeffs, self.b_coeffs)
        axs_final[0].plot(t_fine_plot, phi_pulse_values, 'r-', label='Phase φ(t) (Fourier)')
        
        axs_final[0].set_xlabel(f'Time (L={self.t_final:.2f})')
        axs_final[0].set_ylabel('Phase φ(t) (radians)', color='r')
        axs_final[0].tick_params(axis='y', labelcolor='r')
        axs_final[0].set_title(f'Optimized Pulse (N_terms={self.num_fourier_terms})')
        axs_final[0].set_ylim([0, 2 * np.pi])
        axs_final[0].grid(True)

        ax2 = axs_final[0].twinx()
        time_steps_amp, amp_stair = self.PWC_pulse(self.omega_envelope / self.Omega_max)
        ax2.plot(time_steps_amp, amp_stair, 'g:', label='Amplitude Ω(t)/Ω_max', alpha=0.7)
        ax2.set_ylabel('Normalized Amplitude', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.set_ylim([-0.05, 1.1])
        lines, labels = axs_final[0].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axs_final[0].legend(lines + lines2, labels + labels2, loc='best')


        axs_final[1].clear()
        im = axs_final[1].imshow(fidelities_eval_grid, 
                           extent=[rabi_factors_eval[0], rabi_factors_eval[-1], 
                                   detuning_multipliers_eval[0], detuning_multipliers_eval[-1]], 
                           origin='lower', aspect='auto', cmap='viridis', vmin=0.7, vmax=1.0) 
        fig_final.colorbar(im, ax=axs_final[1], label='Fidelity')
        axs_final[1].set_xlabel('Rabi Frequency Factor (Ω/Ω_ideal)')
        axs_final[1].set_ylabel('Detuning Multiplier (Δ/Ω_max)')
        axs_final[1].set_title(f'Robustness Map (11x11 Eval Grid)\nAvg. Eval Fidelity: {avg_eval_fidelity:.6f}')
        opt_marker_rabis_plot = self.opt_grid_rabi_factors
        opt_marker_detunings_plot = self.opt_grid_detuning_multipliers
        axs_final[1].scatter(np.tile(opt_marker_rabis_plot, len(opt_marker_detunings_plot)), 
                             np.repeat(opt_marker_detunings_plot, len(opt_marker_rabis_plot)),
                             s=50, facecolors='none', edgecolors='red', marker='o', label='Opt. Grid Points')
        axs_final[1].legend(fontsize='small', loc='lower left') 
        axs_final[1].grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        axs_final[1].set_xticks(rabi_factors_eval[::2]) 
        axs_final[1].set_yticks(detuning_multipliers_eval[::2])


if __name__ == '__main__':
    # --- Pulse Parameters ---
    total_pulse_time_L = np.pi * 2.5
    num_slices_for_time_evolution = 200 # More slices for better evolution accuracy with smooth pulse
    fourier_terms_to_optimize = 6
    omega_val = 1.0  
    
    rise_t = total_pulse_time_L * 0.03
    fall_t = total_pulse_time_L * 0.03

    config_optimize_sine_terms = False # Set to False for symmetric pulses (cosine only)
    initial_coeffs_filename = None
    
    grape_fourier_instance = RobustGRAPEFourierCoeffs(
        t_final=total_pulse_time_L, 
        N_slices_for_evolution=num_slices_for_time_evolution,
        num_fourier_terms=fourier_terms_to_optimize,
        omega_max_val=omega_val,
        rise_time=rise_t,
        fall_time=fall_t,
        initial_coeffs_file=initial_coeffs_filename,
        optimize_sine_terms=config_optimize_sine_terms
    )

    fig_envelope, ax_envelope = plt.subplots(figsize=(10, 5), num="Initial Rabi Frequency Envelope (Fourier)")
    time_coords, amp_coords = grape_fourier_instance.PWC_pulse(grape_fourier_instance.omega_envelope)
    ax_envelope.plot(time_coords, amp_coords, 'g-')
    ax_envelope.set_title(f'Pulse Amplitude Envelope Ω(t)\n(rise={rise_t:.2f}, fall={fall_t:.2f})')
    ax_envelope.set_xlabel(f'Time')
    ax_envelope.set_ylabel('Rabi Frequency Ω(t)')
    ax_envelope.set_ylim(0, omega_val * 1.1)
    ax_envelope.grid(True)
    plt.show(block=False)
    plt.pause(0.1)
    
    # --- Optimization Parameters ---
    iterations = 500
    learning_rate = 0.05
    finite_diff_coeff_step = 1e-5 
    
    fig_live_pulse, ax_live_pulse = plt.subplots(figsize=(10, 6), num="Live Fourier Pulse Shape")
    fig_live_fids, ax_live_fids_array = plt.subplots(2, 1, figsize=(10, 8), sharex=True, num="Live Fidelities (Fourier Opt)")
    avg_fidelities_history_opt = [] 

    for i in range(iterations):
        avg_fidelity_opt_grid, _ = grape_fourier_instance.iteration_onestep_numerical_derivative(
            lr=learning_rate, d_coeff=finite_diff_coeff_step
        )
        avg_fidelities_history_opt.append(avg_fidelity_opt_grid) 
        print(f'Iter: {i:4d}, Avg Opt Fidelity (3x3 grid): {avg_fidelity_opt_grid:.6f}')

        if i % 10 == 0 or i == iterations - 1: 
            ax_live_pulse.clear()
            t_plot_live = np.linspace(0, grape_fourier_instance.t_final, 400)
            phi_plot_live = grape_fourier_instance.reconstruct_phi_at_t(
                t_plot_live, grape_fourier_instance.a_coeffs, grape_fourier_instance.b_coeffs
            )
            ax_live_pulse.plot(t_plot_live, phi_plot_live, 'r-', label='Phase φ(t) (Fourier)')
            ax_live_pulse.set_ylabel('Phase φ(t) (radians)', color='r')
            ax_live_pulse.tick_params(axis='y', labelcolor='r')
            ax_live_pulse.set_title(f'Optimizing Fourier Pulse - Iter {i}\nAvg Opt Fid (3x3): {avg_fidelity_opt_grid:.4f}')
            ax_live_pulse.set_ylim([0, 2 * np.pi])
            ax_live_pulse.grid(True)

            ### MODIFICATION START: Overlay amplitude on live plot ###
            ax_amp_live = ax_live_pulse.twinx()
            time_amp_stair, amp_stair = grape_fourier_instance.PWC_pulse(grape_fourier_instance.omega_envelope / omega_val)
            ax_amp_live.plot(time_amp_stair, amp_stair, 'g:', alpha=0.5, label='Amplitude Ω(t)/Ω_max')
            ax_amp_live.set_ylabel('Normalized Amplitude', color='g')
            ax_amp_live.tick_params(axis='y', labelcolor='g')
            ax_amp_live.set_ylim([-0.05, 1.1])
            lines, labels = ax_live_pulse.get_legend_handles_labels()
            lines2, labels2 = ax_amp_live.get_legend_handles_labels()
            ax_live_pulse.legend(lines + lines2, labels + labels2, loc='best')
            ### MODIFICATION END ###
            
            fig_live_pulse.canvas.draw_idle() 
            
            ax_live_fids_array[0].clear(); ax_live_fids_array[1].clear()
            ax_live_fids_array[0].plot(avg_fidelities_history_opt, 'k-', label='Avg. Fidelity (3x3 Opt. Grid)')
            ax_live_fids_array[0].legend(); ax_live_fids_array[0].grid(True)
            ax_live_fids_array[0].set_ylabel('Average Fidelity')
            ax_live_fids_array[0].set_title('Optimization Progress')
            sorted_history = sorted(grape_fourier_instance.individual_fidelities_history.items())
            for name, history in sorted_history:
                if history: ax_live_fids_array[1].plot(history, label=name if i < 20 else None)
            if i < 20: ax_live_fids_array[1].legend(fontsize='xx-small', ncol=3)
            ax_live_fids_array[1].grid(True); ax_live_fids_array[1].set_xlabel('Iteration')
            ax_live_fids_array[1].set_ylabel('Individual Fidelities')
            fig_live_fids.canvas.draw_idle()
            plt.pause(0.01) 

    print("\nFourier Coefficient Optimization Finished.")
    print(f"Final Avg Opt Fidelity (3x3 grid): {grape_fourier_instance.fidelity:.6f}")
    
    grape_fourier_instance.export_fourier_coeffs_to_csv(base_filename="optimized_fourier_coeffs")

    grape_fourier_instance.export_reconstructed_pulse_to_csv(base_filename="reconstructed_fourier_pulse")


    fig_final, axs_final = plt.subplots(1, 2, figsize=(18, 7), num="Final Fourier Pulse and Robustness Map") 
    print("Evaluating on 11x11 grid and plotting final results...")
    grape_fourier_instance.evaluate_and_plot_final_robustness(fig_final, axs_final)
    plt.tight_layout(pad=3.0) 
    plt.show()
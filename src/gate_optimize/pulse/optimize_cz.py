# 20240517 revision, using the Eq.(7) as optimization target
# 
'''
A demo of GRAPE algorithm reproducing Time Optimal Gate by Tao Zhang.
'''

import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import expm
# %matplotlib qt

# optmize CZ gate
class GRAPE():
    def __init__(self):
        # define the Pauli matrices
        self.I = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self.initial_state = np.array([[1],[0],[1],[0]])# 01, 0r, 11, W, this state should NOT be normalized.
        self.target_state = np.array([[1],[0],[-1],[0]])
        self.S01 = np.array([[1],[0],[0],[0]])
        self.S11 = np.array([[0],[0],[1],[0]])
        self.H1 = np.array([[0, 0.5, 0, 0],[0.5, 0, 0, 0],[0, 0, 0, np.sqrt(2)/2],[0, 0, np.sqrt(2)/2, 0]])
        self.H2 = np.array([[0, 0.5j, 0, 0],[-0.5j, 0, 0, 0],[0, 0, 0, 1j*np.sqrt(2)/2],[0, 0, -1j*np.sqrt(2)/2, 0]])
        self.Omega_max = 1

        self.U01 = np.array([[1, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 0, 0]])
        self.U11 = np.array([[0, 0, 0, 0],[0, 0, 0, 0],[0, 0, 1, 0],[0, 0, 0, 0]])
        # time list
        self.t_list = np.linspace(0, 7.6, 100+1) 
        self.dt = self.t_list[1] - self.t_list[0]
        self.phi = np.random.rand(len(self.t_list)) * 2 * np.pi # note that the last phi is actually controlling the single qubit phase gate.
        self.HNplus1 = np.array([[1,0,0,0],[0,0,0,0],[0,0,2,0],[0,0,0,0]]) / self.dt # The amplitude being phi
        self.fidelity = None
        
    def U_indt(self):
        '''
        get the list of single-step evolution operators
        i.e. U(t_n, t_n+1) = exp(-i * H * dt)
        '''
        U_indt_list = []
        for i in range(len(self.t_list)-1):
            U_indt = expm(-1j * self.Omega_max * (np.cos(self.phi[i]) * self.H1 + np.sin(self.phi[i]) * self.H2) * self.dt)
            U_indt_list.append(U_indt)
        U_indt = np.array([[np.exp(-1j*self.phi[-1]),0,0,0],[0,1,0,0],[0,0,np.exp(-2j*self.phi[-1]),0],[0,0,0,1]])
        # H_N+1 is actually 1/Δt * [[phi,0,0,0], [0,0,0,0], [0,0,2*phi,0], [0,0,0,0]], and U_N+1 is [[exp(1j * phi),0,0,0], [0,1,0,0], [0,0,exp(2j * phi),0], [0,0,0,1]], where we do not care what happens to 0r and W state as long as they are not evolved into 10, 01 and 00.
        # This is ensured by the expression of our fidelity, which is something+2 * |a01|^2 + |a11|^2, where the last two terms ensure that no state is leaking out from the subspace where there is no rydberg state.
        U_indt_list.append(U_indt)
        return U_indt_list


    def bidirect_evolu(self, C):

        U_indt_list = self.U_indt()

        rho_list = []
        rho = self.U0
        for i in range(len(self.t_list)):
            rho = U_indt_list[i]@ rho @ U_indt_list[i].conj().T
            rho_list.append(rho)
        
        lambda_list = []
        lambda_ = C
        for i in range(len(self.t_list)):
            lambda_ = U_indt_list[-i-1].conj().T @ lambda_ @ U_indt_list[-i-1]
            lambda_list.append(lambda_)
        lambda_list.reverse()

        return rho_list, lambda_list
    
    def BidirectEvolution(self):
        '''
        Return A_list of Aj where Aj = UN...Uj+1 and B_list of Bj where Bj = Uj...U1
        '''
        U_indt_list = self.U_indt()
        A_list = []
        A = self.I
        for i in range(len(self.t_list)):
            A = A @ U_indt_list[-i-1]
            A_list.append(A)
        A_list.reverse()
        
        B_list = []
        B = self.I
        for i in range(len(self.t_list)):
            B = U_indt_list[i] @ B
            B_list.append(B)

        return A_list, B_list
    

    
    @staticmethod
    def inner_product(A, B):
        '''
        according to Ref. srep36090 (2016),
        the inner product only calculate the trace of A^dagger * B
        '''
        return np.trace(np.dot(A.conj().T, B))

    @staticmethod
    def innerProduct(a, b):
        '''
        Return inner product between <a| and |b>, assuming a and b in column vector form.
        '''
        assert np.shape(a) == np.shape(b) and len(np.shape(a)) == 2 and np.shape(a)[1] == 1
        return (a.conj().T @ b)[0,0]

    @staticmethod
    def inner(a, U, b):
        '''
        Return inner product <a|U|b>, assuming a and b in column vector form and U in square matrix form.
        '''
        assert np.shape(a) == np.shape(b) and len(np.shape(a)) == 2 and np.shape(a)[1] == 1  and len(np.shape(U)) == 2 and np.shape(U)[0] == np.shape(U)[1] and np.shape(U)[0] == np.shape(a)[0]
        return (a.conj().T @ U @ b)[0,0]
    
    def iteration_onestep(self, lr=0.5):
        '''
        iteration of GRAPE algorithm
        '''
        partial_phi = np.zeros_like(self.phi)
        # partial_w0 = np.zeros_like(self.w0)
        # partial_w1 = np.zeros_like(self.w1)

        # Can be accelerated using column vectors as state instead of density matrix & calculate the evolution part only once, then multiplying differentinitial state and final state afterwards
        A_list, B_list = self.BidirectEvolution()
        # rho_list_F, lambda_list_F = self.bidirect_evolu(self.U_F)
        # rho_list_01, lambda_list_01 = self.bidirect_evolu(self.U01)
        # rho_list_11, lambda_list_11 = self.bidirect_evolu(self.U11)
        phi_final = B_list[-1] @ self.initial_state
        a01 = self.innerProduct(self.S01, phi_final)
        a11 = - self.innerProduct(self.S11, phi_final)
        for i in range(len(self.t_list)-1):

            par_a01_par_phi = 1j * self.dt * self.inner(self.S01, A_list[i] @ ( self.H1*np.sin(self.phi[i])-self.H2*np.cos(self.phi[i]) ) @B_list[i], self.initial_state)
            par_a11_par_phi = - 1j * self.dt * self.inner(self.S11, A_list[i] @ ( self.H1*np.sin(self.phi[i])-self.H2*np.cos(self.phi[i]) ) @B_list[i], self.initial_state) # Note the -1 here.

            partial_phi[i] = 1/20 * ( 2 * np.conjugate(par_a01_par_phi) + np.conjugate(par_a11_par_phi) +  2 * par_a01_par_phi + 6 * (par_a01_par_phi * np.conjugate(a01) + a01 * np.conjugate(par_a01_par_phi) ) + 2 * (par_a01_par_phi*np.conjugate(a11) + a01*np.conjugate(par_a11_par_phi)) + par_a11_par_phi + 2 * (par_a11_par_phi*np.conjugate(a01) + a11*np.conjugate(par_a01_par_phi)) + 2 * (par_a11_par_phi*np.conjugate(a11) + a11*np.conjugate(par_a11_par_phi))     )
            
        par_a01_par_phi_Nplus1 = - 1j * self.dt * self.inner(self.S01, self.HNplus1 @ B_list[-1], self.initial_state)
        par_a11_par_phi_Nplus1 = 1j * self.dt * self.inner(self.S11, self.HNplus1 @ B_list[-1], self.initial_state) # Note the -1 here.
        partial_phi[-1] = 1/20 * ( 2 * np.conjugate(par_a01_par_phi_Nplus1) + np.conjugate(par_a11_par_phi_Nplus1) +  2 * par_a01_par_phi_Nplus1 + 6 * (par_a01_par_phi_Nplus1 * np.conjugate(a01) + a01 * np.conjugate(par_a01_par_phi_Nplus1) ) + 2 * (par_a01_par_phi_Nplus1*np.conjugate(a11) + a01*np.conjugate(par_a11_par_phi_Nplus1)) + par_a11_par_phi_Nplus1 + 2 * (par_a11_par_phi_Nplus1*np.conjugate(a01) + a11*np.conjugate(par_a01_par_phi_Nplus1)) + 2 * (par_a11_par_phi_Nplus1*np.conjugate(a11) + a11*np.conjugate(par_a11_par_phi_Nplus1))     )

        # partial_phi[-1] += -2 * np.real(    self.inner_product(lambda_list_F[i], 1j * self.dt * (self.H_Nplus_1 @ rho_list_F[i] - rho_list_F[i] @ self.H_Nplus_1) )   *    self.inner_product(rho_list_F[-1], self.U_F)   ) 
        # partial_phi[-1] += 2 * ( -2 * np.real(    self.inner_product(lambda_list_01[i], 1j * self.dt * (self.H_Nplus_1 @ rho_list_01[i] - rho_list_01[i] @ self.H_Nplus_1) )   *    self.inner_product(rho_list_01[-1], self.U01)   )  )
        # partial_phi[-1] += -2 * np.real(    self.inner_product(lambda_list_11[i], 1j * self.dt * (self.H_Nplus_1 @ rho_list_11[i] - rho_list_11[i] @ self.H1) )   *    self.inner_product(rho_list_11[-1], self.U11)   ) 
        # partial_phi[-1] *= 0
        # print("Final phi:")
        # print(self.phi[-1])

        self.phi = self.phi + lr * partial_phi

        for i_phi in np.arange(len(self.phi)):
            if self.phi[i_phi]>2*np.pi:
                self.phi[i_phi] -= 2*np.pi
            elif self.phi[i_phi]<0:
                self.phi[i_phi] += 2*np.pi
        self.fidelity = 1/20 * (np.abs(1+2*a01+a11)**2 + 1 + 2*np.abs(a01)**2 + np.abs(a11)**2)

        return self.fidelity
    
    def PWC_pulse(self, pwc_pulse):
        '''
        get the piecewise constant pulse, then use plt.plot to plot it
        '''
        pwc_pulse = np.insert(pwc_pulse, 0, 0)
        pwc_pulse = np.append(pwc_pulse, 0)
        time_steps = np.arange(0, len(pwc_pulse)) * self.dt
        time_steps_stair = np.repeat(time_steps, 2)[1:-3]
        pwc_pulse_stair = np.repeat(pwc_pulse, 2)[2:-2]

        return time_steps_stair, pwc_pulse_stair
    
    # def fidelity_crosschecker(self, final_rho):

    

if __name__ == '__main__':
    G = GRAPE()

    infid = []
    i = 0
    lr = 0.5
    plt.figure(0)
    while i < 5001:
        fidelity = G.iteration_onestep(lr)
        # print('{}-th\t fidelity: {:4f}'.format(i, fidelity))
        infid.append(abs(1 - fidelity))
        if i % 10 == 0:
            plt.clf()
            time_steps_stair0, pwc_pulse_stair0 = G.PWC_pulse(G.phi)
            plt.plot(time_steps_stair0, pwc_pulse_stair0, 'b-')
            plt.xlabel('time')
            plt.ylabel('pulse strength')
            plt.title('{}-th fidelity: {:4f}, lr: {:4f}'.format(i, fidelity, lr))
            plt.ylim([0, 2*np.pi])
            plt.pause(0.01)
        if i == 300:
            plt.savefig('pulses_300.png')

        i += 1

    plt.savefig('pulses_final.png')
    plt.show()
    plt.figure(1)
    # use log scale to plot the fidelity
    plt.plot(infid)
    # plt.yscale('log')
    plt.xlabel('iteration')
    plt.ylabel('infidelity')
    plt.savefig('infidelity.png')
    plt.show()
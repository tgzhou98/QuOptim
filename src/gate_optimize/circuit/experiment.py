from .environment import Environment
from prettytable import PrettyTable
from . import utils

import matplotlib.pyplot as plt
import numpy as np
import torch

from .multiproc_env import MultiprocessEnv
from typing import Union
from stim import Tableau

class Experiment:
    """
    Responsible for initializing the environment, agent, and training, evaluating the agent. Sort of an interface for the runner program.
    """
    def __init__(self, algorithm: str, training_req: bool, **kwargs) -> None:
        self.algorithm = algorithm
        self.n_workers = kwargs['n_workers'] if 'n_workers' in kwargs else 8
        self.training_req = training_req

    def initialize_env(self, target_state: Union[list, Tableau], fidelity_tol: float, max_steps: int, gateset: list[str], dist: str, seed: int, **kwargs):
        assert seed, 'seed cannot be None'
        print(seed, 'duhh what', flush=True)
        target_state_tbl = utils.np2tableau(target_state) if not isinstance(target_state, Tableau) else target_state
        self.sample_env = Environment(1, target_state_tbl, fidelity_tol, max_steps, gateset, dist, seed) # dummy env
        if not self.training_req:
            return

        # initialize env
        if self.algorithm == 'ppo':
            env_fn = lambda seed: Environment(kwargs['num_envs'], target_state_tbl, fidelity_tol, max_steps, gateset, dist, seed)
            self.env = MultiprocessEnv(env_fn, seed, self.n_workers)
        else:
            self.env = Environment(1, target_state_tbl, fidelity_tol, max_steps, gateset, dist, seed)
    
    def initialize_test_env(self, start_state: Union[list, Tableau], target_state: Union[list, Tableau], fidelity_tol, max_steps, gateset: list[str], dist: str):
        start_state_tbl = utils.np2tableau(start_state) if not isinstance(start_state, Tableau) else start_state
        target_state_tbl = utils.np2tableau(target_state) if not isinstance(target_state, Tableau) else target_state

        self.sample_env = Environment(1, start_state_tbl, fidelity_tol, max_steps, gateset, dist, None, _start_state=target_state_tbl) # agent prepares in reverse
        return self.sample_env

    def initialize_agent_pg(self, 
                         policy_hidden: tuple,
                         policy_activ_fn,
                         policy_model_max_grad_norm,
                         policy_optimizer_fn,
                         value_hidden: tuple,
                         value_activ_fn, 
                         value_model_max_grad_norm, 
                         value_optimizer_fn, 
                         entropy_loss_weight,
                         gamma):
        if self.algorithm == 'vpg':
            from . import vpg
            policy_model_fn = lambda: vpg.PolicyNeuralNet(    self.sample_env.state_size, policy_hidden, len(self.sample_env.action_space), policy_activ_fn)
            value_model_fn = lambda: vpg.NeuralNet(           self.sample_env.state_size, value_hidden,  1,                          value_activ_fn  )
            self.agent = vpg.VPG(self.sample_env, gamma, 
                        policy_model_fn, policy_model_max_grad_norm, policy_optimizer_fn, 
                        value_model_fn, value_model_max_grad_norm, value_optimizer_fn, entropy_loss_weight)
        elif self.algorithm == 'ppo':
            from . import ppo
            state_dim = self.sample_env.state_size
            print(f'{policy_hidden=}, {policy_activ_fn=}, {state_dim=}, {len(self.sample_env.action_space)=}', flush=True)
            mask = torch.tensor([-0.1 if self.sample_env.gates[act][:4] == 'cnot' or self.sample_env.gates[act][0] == 'h' else -0.03 for act in self.sample_env.action_space], dtype=torch.float32)
            policy_model_fn = lambda: ppo.FCCA(state_dim, len(self.sample_env.action_space), policy_hidden, policy_activ_fn, mask)
            value_model_fn = lambda: ppo.FCV(state_dim, value_hidden, value_activ_fn)
            buf = None
            self.agent = ppo.PPO(
                (self.env if self.training_req else self.sample_env), 
                policy_model_fn, 
                policy_optimizer_fn, 
                value_model_fn, 
                value_optimizer_fn, 
                buf,
                
                policy_model_max_grad_norm=policy_model_max_grad_norm,
                policy_optimization_epochs=8,
                policy_sample_ratio=0.125,
                policy_clip_range=0.2,
                policy_stopping_kl=0.01,
                
                value_model_max_grad_norm=value_model_max_grad_norm,
                value_optimization_epochs=8,
                value_sample_ratio=0.125,
                value_clip_range=float('inf'),
                value_stopping_mse=0.01,
                entropy_loss_weight=entropy_loss_weight,
            )
        else:
            raise ValueError(f'Invalid pg algorithm {self.algorithm}')

    def initialize_agent_vb(self, 
                         hidden: tuple, 
                         activ_fn, 
                         max_grad_norm:float,
                         optimizer_fn, 
                         optimization_epochs,
                         gamma, 
                         training_strategy, 
                         eval_strategy, 
                         polyak: float,
                         rbf_fn, 
                         algo='dq-learn', 
                         dueling=True):
        from . import dqn
        state_dim = self.sample_env.state_size
        model = lambda: dqn.NeuralNet(state_dim, hidden, len(self.sample_env.action_space), activ_fn, dueling)
        rbf = rbf_fn(state_dim)
        self.agent = dqn.DQN(self.sample_env, model, max_grad_norm, optimizer_fn, optimization_epochs, rbf, algo, gamma, polyak, training_strategy, eval_strategy)

    # functions that more or less just collect and present stats over the agent's results
    def train(self, n_eps: int, savepath: str, **kwargs):
        self.savepath = savepath
        if 'start_ep' in kwargs:
            self.start_ep = kwargs['start_ep']
        else:
            self.start_ep = 0
        self.results, eps = self.agent.train(n_eps, plot_fn=self.get_stats, **kwargs) # list of (ep, ter, acts, rew, state, fidelity)
        return self.results, eps
    
    # old function. Unused now.
    def get_stats(self, results, msg=None, roll_ct=100, filesuffix=''):
        if len(results) < 5 * roll_ct:
            print(f'Not enough episodes to plot ({len(results)} < {5*roll_ct}) / {len(results[0])}')
            return
        # to be run during training
        filesuffix = f'_{filesuffix}' if filesuffix else ''
        filename = self.savepath + filesuffix
        self.results = results
        if not isinstance(results[0], list):
            self.results = [results]
        # some preprocessing
        keys = ['n_acts', 'n_dist_acts', 'n_cnots', 'fidel', 'rew', 'n_tdg']
        self.results = [[[
            len(res[2]), # keys[0] 
            res[4], # keys[1]
            sum(self.sample_env.gates[act][:4] == 'cnot' for act in res[2]), # keys[2] 
            res[-1], # keys[3]
            res[3], # keys[4]
            sum(self.sample_env.gates[act][:4] == 'tdg' for act in res[2]) # keys[5]
        ] for res in result] for result in self.results]
        self.results = np.array(self.results).mean(axis=0).T

        start_ep = self.start_ep
        n_actions = self.results[0]
        n_dist_acts = self.results[1]
        n_cnots = self.results[2]
        fidel = self.results[3]
        rews = self.results[4]
        n_tdg = self.results[5]

        # roll_ct = min(roll_ct, self.results.shape[1])
        window = lambda i: slice(i, i+roll_ct)
        xrange = self.results.shape[1] - roll_ct

        rolling_mean_n_actions = [  n_actions[window(i)].mean()   for i in range(xrange)]
        rolling_mean_n_dist_acts = [n_dist_acts[window(i)].mean() for i in range(xrange)]
        rolling_mean_n_cnots = [    n_cnots[window(i)].mean()     for i in range(xrange)]
        rolling_mean_fidel = [      fidel[window(i)].mean()       for i in range(xrange)]
        rolling_mean_rews = [       rews[window(i)].mean()        for i in range(xrange)]
        rolling_mean_n_tdg = [      n_tdg[window(i)].mean()       for i in range(xrange)]

        rolling_max_n_actions = [  min(self.sample_env.max_steps, n_actions[window(i)].mean() + 0.5*n_actions[window(i)].std())     for i in range(xrange)]
        rolling_max_n_dist_acts = [min(self.sample_env.max_steps, n_dist_acts[window(i)].mean() + 0.5*n_dist_acts[window(i)].std()) for i in range(xrange)]
        rolling_max_n_cnots = [    min(self.sample_env.max_steps, n_cnots[window(i)].mean() + 0.5*n_cnots[window(i)].std())         for i in range(xrange)]
        rolling_max_fidel = [      min(1, fidel[window(i)].mean() + 0.5*fidel[window(i)].std())                               for i in range(xrange)]
        rolling_max_rews = [       rews[window(i)].mean() + 0.5*rews[window(i)].std()                                               for i in range(xrange)]
        rolling_max_n_tdg = [      min(self.sample_env.max_steps, n_tdg[window(i)].mean() + 0.5*n_tdg[window(i)].std())           for i in range(xrange)]

        rolling_min_n_actions = [  max(0, n_actions[window(i)].mean() - 0.5*n_actions[window(i)].std())                             for i in range(xrange)]
        rolling_min_n_dist_acts = [max(0, n_dist_acts[window(i)].mean() - 0.5*n_dist_acts[window(i)].std())                         for i in range(xrange)]
        rolling_min_n_cnots = [    max(0, n_cnots[window(i)].mean() - 0.5*n_cnots[window(i)].std())                                 for i in range(xrange)]
        rolling_min_fidel = [      max(0, fidel[window(i)].mean() - 0.5*fidel[window(i)].std())                               for i in range(xrange)]     
        rolling_min_rews = [       rews[window(i)].mean() - 0.5*rews[window(i)].std()                                               for i in range(xrange)]
        rolling_min_n_tdg = [      max(0, n_tdg[window(i)].mean() - 0.5*n_tdg[window(i)].std())                                   for i in range(xrange)]
        
        # plotting
        fig = plt.figure(figsize=(20, 16))
        grid = (2, 2)
        ax1 = plt.subplot2grid(grid, (0, 0))
        ax2 = plt.subplot2grid(grid, (1, 0), colspan=2)
        ax3 = plt.subplot2grid(grid, (0, 1))
        fig.suptitle(f'Results for \n{msg}' if msg is not None else 'Results')
        # set the x-range
        xcoords = np.arange(start_ep, start_ep + xrange)
        # plot rewards
        ax1.plot(xcoords, rolling_mean_rews, 'r-', linewidth=3)
        ax1.axhline(y=0, color='g', linewidth=3)
        ax1.fill_between(xcoords, 0, 1, facecolor='g', alpha=0.1)

        # plot actions
        # ax2.set_yscale('log')
        ax2.plot(rolling_mean_n_actions, 'g-', linewidth=3)
        ax2.plot(rolling_mean_n_dist_acts, 'm-', linewidth=3)
        ax2.plot(rolling_mean_n_cnots, 'c-', linewidth=3)
        ax2.plot(rolling_mean_n_tdg, 'y-', linewidth=3)
        ax2.legend(['number of actions', 'number of actions that increased fidelity', 'number of CX gates', 'number of T gates'])
        
        # plot final fidelity achieved at the end of episodes
        ax3.plot(rolling_mean_fidel, 'b-', linewidth=3)
        
        # labels
        ax1.set_xlabel('Number of epsiodes')
        ax2.set_xlabel('Number of epsiodes')
        ax3.set_xlabel('Number of epsiodes')
        ax1.set_ylabel('Reward')
        ax2.set_ylabel('Number of steps (gates)')
        ax3.set_ylabel('Final fidelity')
        fig.savefig(filename + '_no_mean.png')

        ax1.fill_between(list(range(xrange)), rolling_min_rews, rolling_max_rews, facecolor=[(1, i/xrange, i/xrange) for i in range(xrange)], alpha=0.15)

        ax2.fill_between(list(range(xrange)), rolling_min_n_actions, rolling_max_n_actions, facecolor=[(i/xrange, 1, i/xrange) for i in range(xrange)], alpha=0.15)
        ax2.fill_between(list(range(xrange)), rolling_min_n_dist_acts, rolling_max_n_dist_acts, facecolor=[(1, i/xrange, 1) for i in range(xrange)], alpha=0.15)
        ax2.fill_between(list(range(xrange)), rolling_min_n_cnots, rolling_max_n_cnots, facecolor=[(i/xrange, 1, 1) for i in range(xrange)], alpha=0.15)
        ax2.fill_between(list(range(xrange)), rolling_min_n_tdg, rolling_max_n_tdg, facecolor=[(1, 1, i/xrange) for i in range(xrange)], alpha=0.15)

        ax3.fill_between(list(range(xrange)), rolling_min_fidel, rolling_max_fidel, facecolor=[(i/xrange, i/xrange, 1) for i in range(xrange)], alpha=0.15)
        
        fig.savefig(f'{filename}_with_mean.png')
        fig.clear()

        plt.close('all')
        
    def evaluate(self, env: Union[Environment, None]=None, n_eps: int=1, num_best: int=1, verbose: int=1):
        print("Evaluating", env is not None, flush=True)
        eval_env = env if env is not None else self.sample_env
        print("test:", eval_env.target_state.to_stabilizers(canonicalize=True), flush=True)
        print("test:", eval_env.start_state[0].to_stabilizers(canonicalize=True), flush=True)
        self.results = self.agent.evaluate(eval_env, n_eps)
        print("test:", eval_env.start_state[0].to_stabilizers(canonicalize=True), flush=True)
        eval_results = PrettyTable(['Episode #', 'Number of steps', 'reward', 'final_state', 'fidelity score'])
        eval_results.add_rows([[i, len(acts), rew, *rest] for i, (rew, acts, *rest) in enumerate(self.results)])
        if verbose == 2:
            print(eval_results)
        else: pass
        stats = [[acts, rew, *rest] for rew, acts, *rest in self.results]
        
        return sorted(stats, key=lambda val: (not (val[3]>1-eval_env.tol), len(val[0]) + 1-val[3]))[:num_best], eval_results
        
    def save_model(self, filename):
        with open(f'{filename}.pkl', 'wb') as f:
            if self.algorithm in ['ppo', 'vpg']: 
                torch.save(self.agent.policy_model.state_dict(), f)
            elif self.algorithm in ['dqn', 'ddqn']:
                torch.save(self.agent.online_model.state_dict(), f)
        if self.algorithm in ['vpg', 'ppo']:
            with open(f'{filename}_value.pkl', 'wb') as f:
                torch.save(self.agent.value_model.state_dict(), f)

    def load_model(self, filename):
        with open(f'{filename}.pkl', 'rb') as f:
            if self.algorithm in ['ppo', 'vpg']: 
                self.agent.policy_model.load_state_dict(torch.load(f))
            elif self.algorithm in ['dqn', 'ddqn']:
                self.agent.online_model.load_state_dict(torch.load(f))
        if self.algorithm in ['vpg', 'ppo']:
            with open(f'{filename}_value.pkl', 'rb') as f:
                self.agent.value_model.load_state_dict(torch.load(f))

    def close(self):
        if hasattr(self, 'env'): self.env.close()
        if hasattr(self, 'sample_env'): self.sample_env.close()
        
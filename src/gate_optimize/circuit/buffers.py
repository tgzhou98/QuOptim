import numpy as np
import torch

class ExperienceProcessor:
    """
    Handles all computation required during experience collection in PPO. Run separately by each agent.
    """
    def __init__(self,
                 state_dim,
                 gamma,
                 tau,
                 max_steps,
                 max_steps_per_episode,
                 device,
                 num_envs=16):

        assert max_steps >= max_steps_per_episode

        self.state_dim = state_dim
        self.gamma = gamma
        self.tau = tau
        self.max_steps = max_steps
        self.max_steps_per_episode = max_steps_per_episode # used only for logging
        self.num_envs = num_envs

        self.device = torch.device(device)

        self.states_mem = np.empty(
            shape=np.concatenate(((self.max_steps, self.num_envs), self.state_dim)), dtype=np.float32)
        self.states_mem[:] = np.nan

        self.actions_mem = np.empty(shape=(self.max_steps, self.num_envs), dtype=np.int32)
        self.actions_mem[:] = -1

        self.rewards_mem = np.empty(shape=(self.max_steps, self.num_envs), dtype=np.float32)
        self.rewards_mem[:] = np.nan

        self.returns_mem = np.empty(shape=(self.max_steps, self.num_envs), dtype=np.float32)
        self.returns_mem[:] = np.nan

        self.gaes_mem = np.empty(shape=(self.max_steps, self.num_envs), dtype=np.float32)
        self.gaes_mem[:] = np.nan

        self.logpas_mem = np.empty(shape=(self.max_steps, self.num_envs), dtype=np.float32)
        self.logpas_mem[:] = np.nan

    def fill(self, env, policy_model, value_model):
        
        assert self.num_envs == env.num_envs

        # just logging information. Not used in the optimization
        # information per episode.
        ep_terminated = []
        ep_returns = []
        ep_dists = []
        ep_fidels = []
        ep_acts = []

        self.total_steps = 0
        state, info = env.reset() # note that this initial state will go into the state_mem on the first iteration
        
        gates, _, targets, _, _ = env.prepare_gatelist(env.qubits, env.gateset)
        targets2 = [target if len(target) > 1 else target + target for target in targets]
        targets2 = np.array(targets2, dtype=np.int32)
        penalty = np.ones((len(gates),), dtype=np.float32) * 0.5 / env.max_steps
        for i, gt in enumerate(gates):
            if gt.startswith('sdg') or gt.startswith('hsdgh'):
                penalty[i] *= 1
            else:
                penalty[i] *= 3

        dones = np.zeros((env.num_envs,), dtype=bool)
        rolling_returns = np.zeros((env.num_envs,), dtype=np.float32)
        step_number = np.zeros((env.num_envs,), dtype=np.int32)

        common_shape = tuple(self.actions_mem.shape)
        values = np.zeros(shape=common_shape)
        terminals = np.zeros(shape=common_shape)
        truncateds = np.zeros(shape=common_shape)
        
        prev_step_actions = np.zeros((env.num_envs, env.qubits, env.qubits), dtype=np.int32) - 1
        env_range = np.arange(env.num_envs)

        for timestep in range(self.max_steps):
            # if timestep %10 == 0:print(timestep, flush=True)
            
            # take a step (nowhere here do we need gradients, so torch.no_grad)
            with torch.no_grad():
                action, logpa = policy_model.np_pass(state)
                values[timestep] = value_model.forward(state).cpu().numpy()

            next_state, reward, terminal, truncated, info = env.step(action)
            # print('next_state', next_state, flush=True)
            # print('reward-agent', reward, flush=True)
            step_number += 1

            # # discourage the agent from taking the same action again (not enforced currently)
            # qubits_of_action = targets2[action]
            # # assert qubits_of_action.shape == (env.num_envs, 2), f'{qubits_of_action} {action}'
            # reward -= (prev_step_actions[env_range, qubits_of_action[:, 0], qubits_of_action[:, 1]] == action) * penalty[action]
            # prev_step_actions[env_range, qubits_of_action[:, 0], qubits_of_action[:, 1]] = action
            
            rolling_returns += self.gamma ** (step_number-1) * reward

            terminals[timestep] = terminal
            truncateds[timestep] = truncated
            done = terminal|truncated
            
            ep_returns.extend(rolling_returns[done])
            ep_fidels.extend(env.fidelity_of_resetted)
            ep_dists.extend(env.meta_actions_of_resetted)
            # ep_acts.extend([[2]*steps_taken for steps_taken in step_number[done]])
            ep_acts.extend(step_number[done])
            assert len(ep_acts) == len(ep_returns) == len(ep_dists) == len(ep_fidels)
            rolling_returns[done] = 0
            step_number[done] = 0

            self.states_mem[timestep] = state
            self.actions_mem[timestep]= action
            self.logpas_mem[timestep] = logpa
            self.rewards_mem[timestep]= reward

            state = next_state

        # edge bootstrap
        with torch.no_grad():
            next_value = value_model(state).reshape(1, -1).cpu().numpy()
        assert values.shape[-1] == next_value.shape[-1]

        # now we have a lot of experience. Compute the returns and gaes
        advantages = np.zeros(shape=self.rewards_mem.shape)
        # advantages[-1] = next_value

        assert timestep == self.max_steps - 1
        for t in range(timestep, -1, -1):
            delta = self.rewards_mem[t] + self.gamma * (values[t+1] if t != timestep else next_value) * (1 - terminals[t]) - values[t]
            delta *= (1 - truncateds[t])
            advantages[t] = delta + self.gamma * self.tau * (1 - terminals[t]) * (1 - truncateds[t]) * (advantages[t+1] if t != timestep else 0)

        self.returns_mem = values + advantages
        self.gaes_mem = advantages

        ep_idxs_int = np.arange(len(ep_fidels))
        ep_dists = np.array(ep_dists, dtype=np.int32)
        print('filled', len(ep_fidels), 'episodes total')
        return {'idxs':ep_idxs_int, 'ter':ep_terminated, 'n_steps':ep_acts, 'returns':ep_returns, 'meta':ep_dists, 'fidels':ep_fidels}

    def get_stacks(self) -> tuple[np.array, np.array, np.array, np.array, np.array]:
        return (self.states_mem, self.actions_mem, 
                self.returns_mem, self.gaes_mem, self.logpas_mem)
    
    def update_max_episodes(self):
        # self.max_steps = int(self.max_steps * 1.001)
        # self.clear()
        pass

from .environment import Environment
from .experiment import Experiment
from .color_codes import *
from typing import Union
from functools import partial

from argparse import Namespace
from datetime import datetime
import argparse
import numpy as np
import json
import time
import tqdm
import stim
import os

from . import utils
from . import buffers
import torch
import torch.nn.functional as F
import torch.optim as optim

import cProfile
import pstats
import io

import builtins

from pathlib import Path
from importlib import resources

# --- Add this block at the top ---
# Change the current working directory to the directory of this script
os.chdir(Path(__file__).parent)
# ---------------------------------- 

# Save the original print function
original_print = builtins.print

# Define a new print function that always flushes
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    original_print(*args, **kwargs)

def to_cmd_line(data, encoding:str='dict') -> list:
    s = []
    d = vars(data) if encoding == 'namespace' else data
    for k, v in d.items():
        print(k, v, type(k), type(v), sep='\t\t')
        if v == '': continue
        if isinstance(v, bool):
            if v: s.append(f'--{k}')
            continue
        s.append(f'-{k}')
        s.append(str(v))
    print(s)
    return s

def parse(cmd_line_args: Union[list[str], None] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("-fromjson",type=str,default='')

    parser.add_argument("-name",type=str,default='UNNAMED')
    parser.add_argument("-v",type=int,default=1)
    parser.add_argument("-a",type=str,default="ppo")
    parser.add_argument("-dist",type=str,default='general')

    parser.add_argument("--debug",action='store_true')
    parser.add_argument("--swanlab",action='store_true')
    parser.add_argument("--train",action='store_true')
    parser.add_argument("--ctrain",action='store_true')
    parser.add_argument("-testfile",type=str,default="")

    parser.add_argument("-seed",type=eval,default=[42])
    parser.add_argument("-qbits",type=int,default=2)
    parser.add_argument("-tol",type=float,default=0.1)
    parser.add_argument("-gamma",type=float,default=0.95)
    parser.add_argument("-numeps",type=int,default=10000)
    parser.add_argument("-maxsteps",type=int,default=50)
    parser.add_argument("-rewardtype",type=int,default=0)
    parser.add_argument("-noise",type=float,default=0)
    parser.add_argument("-batchsz",type=int,default=32)
    parser.add_argument("-terct",type=int,default=100)

    # neural net
    parser.add_argument("-phidden",type=eval,default=(256,256))
    parser.add_argument("-vhidden",type=eval,default=(256,256))
    parser.add_argument("-plr",type=float,default=1e-4)
    parser.add_argument("-vlr",type=float,default=1e-4)
    parser.add_argument("-activfn",type=str,default='relu')
    
    # misc hyperparameters
    parser.add_argument("-polyak",type=float,default=5e-3)
    parser.add_argument("-meanbd",type=float,default=1.0)
    parser.add_argument("-stdtol",type=float,default=0.5)

    # ppo specific
    parser.add_argument("-numworkers",type=int,default=8)
    parser.add_argument("-entropywt",type=float,default=1e-4)
    parser.add_argument("-tau",type=float,default=0.97)

    # dqn specific
    parser.add_argument("--duel",action='store_true')
    parser.add_argument("--per",action='store_true')
    parser.add_argument("-expltime",type=float,default=1.0)
    parser.add_argument("-alpha",type=float,default=0.3)
    parser.add_argument("-beta0",type=float,default=0.6)
    parser.add_argument("-betarate",type=float,default=0.9992)
    
    # later
    parser.add_argument("-bufsize",type=int,default=256)
    parser.add_argument("-num_envs",type=int,default=16)
    parser.add_argument("-exptdate", type=str, default='')
    parser.add_argument("-gateset",type=eval,default=['h','cnot','s','hsh'])

    args = parser.parse_args(cmd_line_args)
    if args.fromjson != '':
        with open(args.fromjson, 'r') as f:
            cmd = to_cmd_line(json.load(f), 'dict')
        args = parser.parse_args(cmd)

    if args.exptdate == '':
        print('duhh')
        args.exptdate = datetime.now().strftime('%d-%m-%Y')
    print(f'hi {args=}')
    return args

class Runner:
    """
    Main class to run the experiments. Called directly in runner.__main__, and also in random_testbench.py to run evaluation experiments.
    """
    def __init__(self, args, verbose: int=1) -> None:
        self.args = parse(to_cmd_line(args, encoding='namespace'))
        self.path = Runner.make_path(self.args.qbits, self.args.tol, self.args.name, self.args.exptdate)
        os.makedirs(self.path, exist_ok=True)
        # save to a json
        p = os.path.join(self.path, 'hyper-params.json')
        with open(p, 'w') as f:
            json.dump(vars(args), f, indent=4)

        if self.args.ctrain:
            # load hyperparams from hyper_params.
            if os.path.exists(os.path.join(self.path, 'hyper-params.json')):
                with open(os.path.join(self.path, 'hyper-params.json'), 'r') as f:
                    self.args = json.load(f)
                self.args = Namespace(**self.args)
            else:
                print('no hyper-params.json found')
                exit()
            self.args.ctrain = True

        self.verbose = verbose
        self.target_state = stim.Tableau(self.args.qbits)
        utils._globals['debug'] = self.args.debug
        utils._globals['dist'] = self.args.dist
        utils._globals['rewardtype'] = self.args.rewardtype
        utils._globals['swanlab'] = self.args.swanlab
        utils._globals['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        utils._globals['noise'] = (lambda state: state) if abs(self.args.noise) < 1e-6 else partial(utils.dephasing_noise, prob=self.args.noise)
        utils._globals['bufsize'] = args.bufsize
        utils._globals['gamma'] = args.gamma
        utils._globals['tau'] = args.tau
        utils._globals['num_envs'] = args.num_envs
        print('globals:\n', utils._globals)
        utils.args = args
    
    @staticmethod
    def make_path(qbits: int, tol: float, name: str, exptdate: str) -> str:
        try:
            project_root = resources.files('gate_optimize').parent.parent
        except (ImportError, AttributeError):
            # Fallback for older Python or if resources fails
            project_root = Path(__file__).resolve().parent.parent.parent.parent
        plotdir = str(project_root / 'model' / 'plots')
        # if exptdate != '':
        #     date = exptdate
        # else:
        #     date = datetime.now().strftime('%d-%m-%Y')
        id = f'{qbits}-{tol}-{name}--{exptdate}'
        return os.path.join(plotdir, id)
    
    def setup(self):
        self.policy_optimizer_fn = lambda net: optim.Adam(net.parameters(), lr=self.args.plr)
        self.value_optimizer_fn  = lambda net: optim.Adam(net.parameters(), lr=self.args.vlr)
        self.activfn = getattr(F, self.args.activfn)
        # self.path = Runner.make_path(self.args.qbits, self.args.tol, self.args.name, self.args.exptdate)
        os.makedirs(self.path, exist_ok=True)
        with open(os.path.join(self.path, 'hyper-params.txt'), 'w') as f:
            f.write(str(self.args) + '\n')
        # write the command used to run this experiment to a file
        with open(os.path.join(self.path, 'command.txt'), 'w') as f:
            f.write('python ')
            for arg, val in vars(self.args).items():
                if isinstance(val, bool):
                    if val: f.write(f'--{arg} ')
                elif isinstance(val, list):
                    f.write(f'-{arg} {tuple(val)} ')
                else:
                    f.write(f'-{arg} {val} ')
            f.write('\n')

    def make_expt(self):
        print(f'{self.seed=}')
        self.exp = Experiment(self.args.a, self.training_req, n_workers=self.args.numworkers)
        if self.training_req:
            self.exp.initialize_env(self.target_state, self.args.tol, self.args.maxsteps, self.args.gateset, self.args.dist, self.seed, training_req=self.training_req, n_workers=self.args.numworkers, num_envs=utils._globals['num_envs'])
        
    def initialize_agent(self):
        if self.args.a in ['ppo', 'vpg']:
            self.exp.initialize_agent_pg(
                policy_hidden=self.args.phidden,
                policy_activ_fn=self.activfn,
                policy_model_max_grad_norm=0.5,#float('inf'),
                policy_optimizer_fn=self.policy_optimizer_fn,
                value_hidden=self.args.vhidden,
                value_activ_fn=self.activfn,
                value_model_max_grad_norm=0.5,
                value_optimizer_fn=self.value_optimizer_fn,
                entropy_loss_weight=self.args.entropywt,
                gamma=self.args.gamma,
            )
        else:
            training_strategy = lambda: utils.ExpDecEpsGreedyStrategy(1.0, 0.1, decay_steps=(self.args.expltime * self.args.numeps))
            evaluation_strategy = lambda: utils.EpsGreedyStrategy(epsilon=0)
            bufsize = 50 * self.args.batchsz # can make a hyperparameter
            rbf_fn = lambda statesize: buffers.PrioritizedReplayBuffer(m_size=bufsize, batch_size=self.args.batchsz, statesize=statesize, alpha=self.args.alpha, beta0=self.args.beta0, beta_rate=self.args.betarate, replace=False)
            algo = 'dq-learn'
            self.exp.initialize_agent_vb(
                hidden=self.args.vhidden,
                activ_fn=self.activfn,
                max_grad_norm=1,
                optimizer_fn=self.value_optimizer_fn,
                optimization_epochs=40,
                gamma=self.args.gamma,
                training_strategy=training_strategy,
                eval_strategy=evaluation_strategy,
                rbf_fn=rbf_fn,
                polyak=self.args.polyak,
                algo=algo,
                dueling=self.args.duel
            )

    def train_agent(self):
        print(RED + '!!checks!!')
        print(hasattr(self.exp, 'env'))
        print(type(self.exp.env))
        # exit()
        print(BLUE + '=========Training starts here=========' + RESET)
        print(os.path.join(self.path, 'results'))
        results, self.curr_ep = self.exp.train(self.args.numeps, savepath=os.path.join(self.path, 'results'), roll_ct=self.args.terct, mean_bound=self.args.meanbd, std_tol=self.args.stdtol, start_ep=self.curr_ep)
        # self.exp.get_stats(os.path.join(self.path, 'results'), roll_ct=50)
        with open(os.path.join(self.path, 'metadata.txt'), 'w') as f: f.write(str(self.curr_ep))
        self.exp.save_model(os.path.join(self.path, 'model'))
        print(BLUE + '=========Training ends here=========' + RESET)
        print(GREEN+'model in', os.path.join(self.path, 'model.pkl'), RESET)
        return results
        
    def test_agent(self, env: Environment) -> list:
        # print('start', env.start_state)
        # print('sv', env.start_state[0].to_state_vector())
        tries = 5
        best, _ = self.exp.evaluate(env, n_eps=tries, num_best=tries, verbose=self.verbose)
        if self.verbose > 0:
            # print(best[0][0])
            # print(self.exp.sample_env.get_inverted_ckt(best[0][0]))
            fid = best[0][3]
            s = "shortest ckt, fidelity: " + (GREEN if fid > 1-self.args.tol else RED) + f'{fid:.4f} ' + RESET + f'(gates = {len(best[0][0])}, basic_gates = {env.num_basic_gates(best[0][0])}\n'
            print('best', [env.gates[a.item()] for a in best[0][0]])
            best_fidel = best[0]
            for ckt in best[1:]:
                if ckt[3] > best_fidel[3] + 1e-6 or (abs(ckt[3] - best_fidel[3]) < 1e-6 and len(ckt[0]) < len(best_fidel[0])): best_fidel = ckt
            s += "best ckt, fidelity:     " + (GREEN if best_fidel[3] > 1-self.args.tol else RED) + f'{best_fidel[3]:.4f} ' + RESET + f'(gates = {len(best_fidel[0])})\n'
            print(s + '-'*250)
        return best
    
    def main(self) -> Union[list, None]:
        self.setup()
        print('setup complete')
        if isinstance(self.args.seed, int):
            # print('ok')
            self.args.seed = [self.args.seed]
        train_results = []
        test_results = []
        self.training_req = self.args.train or self.args.ctrain
        print(f'{self.training_req=}')
        if self.training_req and utils._globals['swanlab']:
            import swanlab
            swanlab.init(project='qsp-rl', name=self.path[6:], config=vars(self.args))
        try:
            for seed in self.args.seed:
                self.seed = seed
                utils.set_seed(seed=self.seed)
                self.make_expt()
                print('env made', flush=True)
                if self.training_req: 
                    print('training required', flush=True)
                    self.initialize_agent()
                    print('agent initialized', flush=True)
                self.curr_ep = 0
                if self.args.ctrain:
                    self.exp.load_model(os.path.join(self.path, 'model'))
                    with open(os.path.join(self.path, 'metadata.txt'), 'r') as f: self.curr_ep = int(f.read())
                    train_results.append(self.train_agent())
                elif self.args.train:
                    print('training', flush=True)
                    train_results.append(self.train_agent())
                    print('training done', flush=True)
                else:
                    utils.debug()
                    kwargs = {}
                    if self.args.dist.startswith('bounded'): 
                        kwargs['depth'] = int(self.args.dist[7:])
                        # kwargs['gateset'] = list(Environment.prepare_gatelist(self.args.qbits, self.args.dist)[-1].values()) ### CURRENTLY NOT SUPPORTED BECAUSE gatelist does not return the inverse circuits
                        print(kwargs['gateset'])
                    elif self.args.dist.startswith('clifford-brickwork'):
                        kwargs['depth'] = int(self.args.dist[18:])

                    test_set = utils.prepare_testbench_tableau(self.args.numeps, self.args.qbits, self.args.testfile, self.seed, overwrite=False, dist=self.args.dist, **kwargs)
                    results = []
                    for target in tqdm.tqdm(test_set):
                        # print('START', self.target_state)
                        # print('TARGET', target)
                        env = self.exp.initialize_test_env(self.target_state, target, self.args.tol, self.args.maxsteps, self.args.gateset, self.args.dist)
                        env.max_steps = int(1 * self.args.maxsteps)
                        if not hasattr(self.exp, 'agent'):
                            self.initialize_agent()
                            mfile = os.path.join(self.path, 'model')
                            try:
                                self.exp.load_model(mfile)
                            except: print('failed to load model for testing')
                        results.append(self.test_agent(env))
                    test_results.append(results)
                utils.debug()
                self.exp.close()
        except Exception as e:
            print('oof', e)
            import traceback
            traceback.print_tb(e.__traceback__)
            # save models
            if self.training_req:
                self.exp.save_model(os.path.join(self.path, 'model'))
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            import IPython
            IPython.embed()
        finally:
            # save results to files
            with open(os.path.join(self.path, 'train-results.txt'), 'a') as f:
                f.write(str(train_results) + '-'*50 + '\n')
            with open(os.path.join(self.path, 'test-results.txt'), 'a') as f:
                f.write(str(test_results) + '-'*50 + '\n')
            if self.training_req:
                print([len(res) for res in train_results])
                return
            else:
                return test_results

if __name__ == '__main__':

    float_formatter = '{:.2f}'.format
    np.set_printoptions(formatter={'complexfloat':float_formatter})
    print('hi')
    args = parse(None)
    print(args)
    start_time = time.time()
    print(start_time)
    pr = cProfile.Profile()
    pr.enable()

    Runner(args, args.v).main()
    
    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(1000)
    print(s.getvalue()[:8000])
    print('total time taken:', f'{time.time()-start_time:.0f} seconds')
    print('testing...')
    
    # Import and run random_testbench instead of using shell command
    try:
        import sys
        print(sys.executable)
        from . import random_testbench
        
        # Call the test function directly with the required parameters
        hyp_string = f'{args.qbits},{args.tol},{args.name},{args.exptdate}'
        test_name = f'random-test-{args.qbits}q'
        
        random_testbench.test(
            params=hyp_string,
            n=50,
            test_name=test_name,
            seed=42,
            dist='clifford',
            verbose=args.v,
            just_qiskit=0
        )
        
    except ImportError as e:
        print(f'Could not import random_testbench: {e}')
    except Exception as e:
        print(f'Error running random_testbench: {e}')
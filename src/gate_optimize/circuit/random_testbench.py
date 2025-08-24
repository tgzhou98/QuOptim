from argparse import ArgumentParser, Namespace
from .environment import Environment
from .runner import Runner
import matplotlib.pyplot as plt
from . import utils
import json
import tqdm
import os

import qiskit.quantum_info as qi
import qiskit.synthesis as qs
from qiskit import QuantumCircuit

def bravyi_circuit(stabilizer_generators: list[str]) -> QuantumCircuit:
    return qi.StabilizerState.from_stabilizer_list(stabilizer_generators).clifford.to_circuit()

def gen_all(params: str, n, test_name, seed, dist, verbose=0):
    q, tol, name, date = params.split(',')
    params_path = Runner.make_path(q, tol, name, date)
    exists_json = os.path.exists(os.path.join(params_path, 'hyper-params.json'))
    if False:#not exists_json:
        params_path = os.path.join(params_path, 'hyper-params.txt')
        with open(params_path) as f:
            args: Namespace = eval(f.read())
    else:
        # print('exists')
        params_path = os.path.join(params_path, 'hyper-params.json')
        with open(params_path) as f:
            args = json.load(f)

    old_args = Namespace(**vars(args))
    q = args.qbits
    dist = dist if dist is not None else args.dist
    args.ctrain = False
    args.train = False
    args.numeps = n
    args.testfile = test_name
    args.dist = dist
    args.seed = seed
    args.fromjson = False
    args.exptdate = date
    # print(f'{args=}')
    runner = Runner(args, verbose)
    results = runner.main()
    savepath = os.path.join(runner.path, 'testbench-results')
    # print(f'{savepath=}')
    collect_stats(args.qbits, results[0], savepath, args.tol)
    # put back the original args
    args = old_args
    with open(os.path.join(runner.path, 'hyper-params.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

def test(params: str, n, test_name, seed, dist, verbose=0, just_qiskit=0):
    q, tol, name, date = params.split(',')
    params_path = Runner.make_path(q, tol, name, date)
    # print(f'{params_path=}')
    if not just_qiskit:
        exists_json = os.path.exists(os.path.join(params_path, 'hyper-params.json'))
        # print(f'{exists_json=}')
        if not exists_json:
            params_path = os.path.join(params_path, 'hyper-params.txt')
            with open(params_path) as f:
                args: Namespace = eval(f.read())
        else:
            # print('exists')
            params_path = os.path.join(params_path, 'hyper-params.json')
            with open(params_path) as f:
                args = json.load(f)
            args = Namespace(**args)

        old_args = Namespace(**vars(args))
        q = args.qbits
        dist = dist if dist is not None else args.dist
        args.ctrain = False
        args.train = False
        args.numeps = n
        args.testfile = test_name
        args.dist = dist
        args.seed = seed
        args.fromjson = ''
        args.exptdate = date
        # print(f'{args=}')
        runner = Runner(args, verbose)
        # exit()
        results = runner.main()
        savepath = os.path.join(runner.path, 'testbench-results')
        # print('yeah from test')

        # put back old args
        args = old_args
        with open(os.path.join(runner.path, 'hyper-params.json'), 'w') as f:
            json.dump(vars(args), f, indent=4)
        
        # plot-plot-plot
        collect_stats(args.qbits, args.gateset, results[0], savepath, args.tol)

    acts_ag = []
    acts_bm = []
    acts_bm_greedy = []
    def stabs_to_cliff(stabs):
        cl = qi.StabilizerState.from_stabilizer_list(stabs).clifford
        return cl
    
    fidels = []
    generators_bigidx_first = []
    i = 0
    with open(test_name) as f:
        for line in tqdm.tqdm(f):
            i += 1
            # if i > 10: break
            line = eval(line.split('#')[0].rstrip())
            line = [l.replace('_', 'I') for l in line]
            # print(line)
            stabilizer_generators = [gentr[0] + ''.join(reversed(gentr[1:])) for gentr in line]
            generators_bigidx_first.append(stabilizer_generators)
            cliff = stabs_to_cliff(stabilizer_generators)

            acts_ag.append(sum(qs.synth_clifford_ag(cliff).count_ops().values()))
            acts_bm.append(sum(qs.synth_clifford_full(cliff).count_ops().values()))
            acts_bm_greedy.append(sum(qs.synth_clifford_greedy(cliff).count_ops().values()))
            # print('gates =', sum(acts[-1].values()))
            fidels.append(1.0)
    # print(acts)
    # print('A-G')
    # print('average number of basic gates: ', sum(acts_ag)/len(acts_ag))
    # print('median number of basic gates: ', sorted(acts_ag)[len(acts_ag)//2])
    # print('min basic gates', min(acts_ag))
    # print('max basic gates', max(acts_ag))
    # print('acts', acts_ag)
    # print('B-M')
    # print('average number of basic gates: ', sum(acts_bm)/len(acts_bm))
    # print('median number of basic gates: ', sorted(acts_bm)[len(acts_bm)//2])
    # print('min basic gates', min(acts_bm))
    # print('max basic gates', max(acts_bm))
    # print('acts', acts_bm)
    # print('B-M-Greedy')
    # print('average number of basic gates: ', sum(acts_bm_greedy)/len(acts_bm_greedy))
    # print('median number of basic gates: ', sorted(acts_bm_greedy)[len(acts_bm_greedy)//2])
    # print('min basic gates', min(acts_bm_greedy))
    # print('max basic gates', max(acts_bm_greedy))
    # print('acts', acts_bm_greedy)

    savepath = Runner.make_path(q, tol, name, date)
    plot_actions_and_fidels(acts_ag, fidels, os.path.join(savepath, 'testbench-results-ag'))
    plot_actions_and_fidels(acts_bm, fidels, os.path.join(savepath, 'testbench-results-bm'))
    plot_actions_and_fidels(acts_bm_greedy, fidels, os.path.join(savepath, 'testbench-results-bm-greedy'))

    if not just_qiskit:
        print_circuits(args.qbits, args.gateset, results[0][:10], generators_bigidx_first[:10])

    print_entanglement_entropy(args.qbits, args.gateset, results[0])

def print_entanglement_entropy(qbits: int, gateset: list[str], results: list):
    gates, ckts, targets, _, basic_gates_count = Environment.prepare_gatelist(qbits, gateset)
    import stim
    inv_ckts = [ckt.inverse() for ckt in ckts]
    ent_entrs = []
    for res in results:
        # print(res)
        acts = res[0][0]
        # print(acts)
        tab = stim.Tableau(qbits)
        ent_entr = []
        for a in reversed(acts):
            tab.append(inv_ckts[a], targets[a])
            entr = utils.ent_entr(tab)
            ent_entr.append(entr)
        ent_entrs.append(ent_entr)
    # print(ent_entrs)
    return ent_entrs

def print_circuits(qbits: int, gateset: list[str], results: list, generators: list):
    gates, _, targets, _, basic_gates_count = Environment.prepare_gatelist(qbits, gateset)
    qc_hsdgh = QuantumCircuit(1)
    qc_hsdgh.h(0)
    qc_hsdgh.s(0)
    qc_hsdgh.h(0)
    qc_hsdgh = qc_hsdgh.to_gate(label='HSH')
    # print(gates)
    for res, gens in zip(results, generators):
        # print(res)
        acts = res[0][0]
        # print(acts)
        # print('gens', gens)
        qc = QuantumCircuit(qbits)
        for i in reversed(acts):
            # print('gates', gates[i], targets[i])
            if gates[i].split('(')[0] == 'h':
                qc.h(targets[i])
            elif gates[i].split('(')[0] == 'cnot':
                qc.cx(*targets[i])
            elif gates[i].split('(')[0] == 'sdg':
                qc.s(targets[i])
            elif gates[i].split('(')[0] == 'hsdgh':
                qc.append(qc_hsdgh, [targets[i]])
            elif gates[i].split('(')[0] == 'x':
                qc.x(targets[i])
            elif gates[i].split('(')[0] == 'z':
                qc.z(targets[i])
            elif gates[i].split('(')[0] == 'y':
                qc.y(targets[i])
            elif gates[i].split('(')[0] == 'tdg':
                qc.t(targets[i])
        # print('rl')
        # print(qc)
        # print('qiskit')
        qc_qiskit = bravyi_circuit(gens)
        # print(qc_qiskit)
        # check if the state is same as final state or not
        sv_rl = qi.Statevector.from_instruction(qc).data
        sv_qiskit = qi.Statevector.from_instruction(qc_qiskit).data
        # print('fidel', qi.state_fidelity(sv_rl, sv_qiskit))
        fid = qi.state_fidelity(sv_rl, sv_qiskit)
        # if not abs(1 - fid) < 1e-6:
        #     print('prepfailed', fid)

def collect_stats(qbits: int, gateset: list[str], results: list, savepath: str, fidel_tol):
    # results: list of (list of (acts, rew, final_state, fidel) over runs) over test cases
    avg_acts, avg_rew, avg_final_state, avg_fidel = [], [], [], [] # called avg but actually best over runs (independently. We are only testing with 1 run here so no difference)
    successes_shortest = 0
    successes_best = 0
    gates, _, targets, _, basic_gates_count = Environment.prepare_gatelist(qbits, gateset)
    def num_basic_gates(acts: list[int]):
        return len(acts)
        ans = 0
        for i in range(len(acts)):
            ans += basic_gates_count[acts[i]]
            if gates[acts[i]].split('(')[0] != 'hsdgh': continue
            if targets[acts[i]] == targets[acts[i-1]] and gates[acts[i-1]].split('(')[0] == 'h':
                ans -= 2; continue
            if i+1 < len(acts) and targets[acts[i]] == targets[acts[i+1]] and gates[acts[i+1]].split('(')[0] == 'h':
                ans -= 2; continue
        return ans
    
    # compute average over runs
    for res in results:
        # print('helloworld', res, len(res), flush=True)
        # for r in res:
        #     print('actsz', len(r[0]), r[1], r[2], r[3], flush=True)
        # avg_acts.append(min([len(r[0]) for r in res]))
        avg_acts.append(min([num_basic_gates(r[0]) for r in res]))
        avg_rew.append(max([r[1] for r in res]))
        # avg_final_state.append(max([r[2] for r in res])) # variation in final state is probably more important?
        avg_fidel.append(max([r[3] for r in res]))
        shortest_idx = 0
        for i in range(len(res)):
            if len(res[i][0]) < len(res[shortest_idx][0]): shortest_idx = i
        successes_shortest += (1 if res[shortest_idx][3] >= 1 - fidel_tol else 0)
        best_fidel_idx = 0
        for i in range(len(res)):
            if res[i][3] > res[best_fidel_idx][3]: best_fidel_idx = i
        best_fidel = res[best_fidel_idx][3]
        successes_best += (1 if best_fidel >= 1 - fidel_tol else 0)
        # assert shortest_idx == best_fidel_idx == 0 # for now
    # print('average number of basic gates: ', sum(avg_acts)/len(avg_acts))
    # print('median number of basic gates: ', sorted(avg_acts)[len(avg_acts)//2])
    # print('min basic gates', min(avg_acts))
    # print('max basic gates', max(avg_acts))
    # print('acts', avg_acts)
    # print('avg reward', sum(avg_rew)/len(avg_rew))
    # print('shortest gate success rate:', f'{successes_shortest}/{len(results)}')
    # print('best fidelity success rate:', f'{successes_best}/{len(results)}')

    # print the various quantities to a file
    with open(savepath + '.txt', 'w') as f:
        f.write(f'number of basic gates: {avg_acts}\n')
        f.write(f'reward: {avg_rew}\n')
        f.write(f'fidelity: {avg_fidel}\n')
        f.write(f'final state: {avg_final_state}\n')
        f.write(f'shortest gate success rate: {successes_shortest}/{len(results)}\n')
        f.write(f'best fidelity success rate: {successes_best}/{len(results)}\n')

    plot_actions_and_fidels(avg_acts, avg_fidel, savepath)
    # now smoothen this data
    # smooth_window = 50
    # avg_acts = utils.smoothen(avg_acts, smooth_window)[:-1]
    # avg_rew = utils.smoothen(avg_rew, smooth_window)[:-1]
    # avg_final_state = utils.smoothen(avg_final_state, smooth_window)[:-1]
    # avg_fidel = utils.smoothen(avg_fidel, smooth_window)[:-1]

    # # plot
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axes[0].plot(avg_acts)
    # axes[0].set_title('Average number of actions')
    # axes[1].plot(avg_fidel)
    # axes[1].set_title('Average fidelity')
    # plt.savefig(savepath + '.png')

def plot_actions_and_fidels(acts, fidels, savepath):
    # plot histogram of the gates and fidelities
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.hist(acts, bins=50, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
    plt.title('Number of actions')
    plt.savefig(savepath + '-acts.png')
    plt.clf()
    plt.hist(fidels, bins=50, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5)
    plt.title('Fidelity')
    plt.savefig(savepath + '-fidel.png')
    plt.clf()
if __name__ == '__main__':
    # print('crazy stuff')
    parser = ArgumentParser()
    parser.add_argument('-n', type=int, default=10)
    parser.add_argument('-seed', type=int, default=42)
    parser.add_argument('-hyp', type=str, default=None)
    parser.add_argument('-name', type=str, default='')
    parser.add_argument('-v', type=int, default=1)
    parser.add_argument('-just-gen', type=int, default=0)
    parser.add_argument('-just-qiskit', type=int, default=0)
    parser.add_argument("-dist",type=str,default='clifford')
    args = parser.parse_args()
    if args.just_gen:
        kwargs = {}
        if args.dist.startswith('clifford-brickwork'):
            kwargs['depth'] = int(args.dist[18:])
        utils.prepare_testbench_tableau(args.n, args.just_gen, args.name, args.seed, True, dist=args.dist, **kwargs)
    if args.hyp:
        test(args.hyp, args.n, args.name, args.seed, args.dist, args.v, args.just_qiskit)

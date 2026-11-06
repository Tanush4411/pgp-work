"""Shared stub world for testing the patched network.py + simulate_new.print_fairness_metrics + sweep_final.parse_log"""
import sys, os, types, importlib.util, io, contextlib, math
PATCH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REPO = os.path.abspath(os.environ.get("SHARDEVAL_REPO", PATCH))
import numpy as np
from unittest.mock import MagicMock

def mod(name, **attrs):
    m = types.ModuleType(name); m.__path__ = []
    for k, v in attrs.items(): setattr(m, k, v)
    sys.modules[name] = m; return m

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m

class Block:
    def __init__(self, id, transactions_list, params):
        self.id, self.transactions_list, self.params = id, transactions_list, params
class Consensus:
    def __init__(self, mu, sigma): self.mu, self.sigma = mu, sigma
    def get_consensus_time(self):
        x = np.random.normal(self.mu, self.sigma)
        while x < 0: x = np.random.normal(self.mu, self.sigma)
        return x + 1
    def get_consensus_probability(self): return 1 - 0.5 * (1 + math.erf((0.5 - self.mu) / self.sigma / math.sqrt(2)))

mod("nodes"); mod("nodes.participating_node", ParticipatingNode=object); mod("nodes.full_node", FullNode=object)
mod("utils"); mod("utils.spanning_tree", SpanningTree=object)
mod("utils.helper", assign_next_hop_to_leader=None, get_transmission_delay=lambda a, b: 0.15)
mod("utils.color_print", ColorPrint=MagicMock())
mod("network"); mod("network.pipe", Pipe=object); mod("network.broadcast", broadcast=None)
mod("network.block", Block=Block); mod("network.consensus"); mod("network.consensus.consensus", Consensus=Consensus)
mod("simpy")
load("network.shard_scheduling", os.path.join(REPO, "network", "shard_scheduling.py"))
load("network.joint_shard_block", os.path.join(REPO, "network", "joint_shard_block.py"))
netmod = load("network.network", os.path.join(PATCH, "network", "network.py"))
simmod = load("simulate_new", os.path.join(PATCH, "simulate.py"))
sweep = load("sweep_final", os.path.join(PATCH, "sweep_final.py"))

class Env:
    def __init__(self): self.now = 0.0
    def timeout(self, d): return d
    def process(self, gen): self.captured = gen

class Tx:
    def __init__(self, id, t, value, target):
        self.id, self.creation_time, self.value, self.target_shard_id, self.cross_shard_status = id, t, value, target, 1

class Queue:
    def __init__(self): self.items = []
    def length(self): return len(self.items)
    def get(self, n): return list(self.items[:n])
    def pop(self, n): out = self.items[:n]; del self.items[:n]; return out

class Pool:
    def __init__(self): self.cross_shard_tx_queues = {}
    def put(self, tx): self.cross_shard_tx_queues.setdefault(tx.target_shard_id, Queue()).items.append(tx)
    def get_cross_shard_pending_count(self, t): q = self.cross_shard_tx_queues.get(t); return q.length() if q else 0
    def get_transaction(self, cap, typ, target_shard_id=None):
        q = self.cross_shard_tx_queues.get(target_shard_id); return q.get(cap) if q else []
    def pop_transaction(self, cnt, typ, target_shard_id=None):
        q = self.cross_shard_tx_queues.get(target_shard_id); return q.pop(cnt) if q else []

class Node:
    def __init__(self, id, leader_id, params, shard):
        self.id, self.shard_leader_id, self.params, self.shard_id = id, leader_id, params, shard
        self.location, self.transaction_pool = "Tokyo", Pool()
    def update_blockchain(self, block): self.params.setdefault("chain", []).append(block)
    def cast_vote_for_joint_shard_block(self, jb):
        p = Consensus(1, 1).get_consensus_probability()
        for tx in jb.transactions_list:
            jb.votes_status[tx.id][self.id] = 1 if np.random.random() < p else 0

def build(params, n_shards, nodes_per_shard):
    env = Env(); net = object.__new__(netmod.Network)
    net.env, net.params, net.full_nodes, net.shard_nodes = env, params, {}, []
    for s in range(n_shards):
        ids = [f"S{s}N{k}" for k in range(nodes_per_shard)]
        for i in ids: net.full_nodes[i] = Node(i, ids[0], params, s)
        net.shard_nodes.append(ids)
    for name in ("partition_nodes", "establish_network_connections", "allow_transactions_generation", "display_network_info"):
        setattr(net, name, lambda: None)
    return net, env

def make_arrivals(n, rate, sim_time, skew, seed):
    rs = np.random.RandomState(1000 + seed)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    order = rs.permutation(len(pairs)); w = np.zeros(len(pairs))
    for rank, idx in enumerate(order, 1): w[idx] = 1.0 / rank ** skew if skew > 0 else 1.0
    w /= w.sum(); out, t, k = [], 0.0, 0
    while True:
        t += rs.exponential(1.0 / rate)
        if t >= sim_time: break
        a, b = pairs[rs.choice(len(pairs), p=w)]
        if rs.rand() < 0.5: a, b = b, a
        out.append((t, a, b, k)); k += 1
    P = None
    if skew > 0:
        W = np.zeros((n, n))
        for idx, (i, j) in enumerate(pairs): W[i][j] = W[j][i] = w[idx]
        P = (W / W.sum(axis=1, keepdims=True)).tolist()
    return out, P

def run_mode(label, n, skew, seed, rate=2.0, sim_time=1000, weighted=False, dynamic=False, alpha=0.1, k=0,
             extras=None, nodes_per_shard=None, warmup=100.0):
    arrivals, P = make_arrivals(n, rate, sim_time, skew, seed)
    params = dict(num_shards=n, verbose=0, merge_capacity=10, elastic_capacity_enabled=True, cutoff_vote_percentage=0.5,
                  simulation_time=sim_time, warmup_time=warmup, weighted_scheduling=weighted, dynamic_matching=dynamic,
                  aging_alpha=alpha, info_delay_rounds=k, sharon_enabled=True, traffic_matrix=P, traffic_skew=skew,
                  processed_tx_count=0, processed_cross_shard_tx_count=0, generated_cross_shard_tx_count=0,
                  generated_tx_count=0, generated_intra_shard_tx_count=0, processed_intra_shard_tx_count=0)
    params.update(extras or {})
    net, env = build(params, n, nodes_per_shard or max(3, 65 // n))
    np.random.seed(seed)                                   # global RNG: consensus + votes (differs by mode, like the real sim)
    net.run_epoch()
    gen = net.run_sharon_cross_shard_phase()
    params["tx_start_time"] = 0.0
    t, idx = 0.0, 0
    while t < sim_time:
        env.now = t
        while idx < len(arrivals) and arrivals[idx][0] <= t:
            at, a, b, kk = arrivals[idx]
            net.get_shard_leader(a).transaction_pool.put(Tx(f"T{kk}", at, 50, b))
            params["generated_cross_shard_tx_count"] += 1; idx += 1
        t += next(gen)
    params["chain"] = params.get("chain", [])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        simmod.print_fairness_metrics(net, params)
    row = sweep.parse_log_text(buf.getvalue()) if hasattr(sweep, "parse_log_text") else None
    return net, params, buf.getvalue()

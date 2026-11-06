import sys, os, types, importlib.util, heapq, random
PATCH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
REPO = os.path.abspath(os.environ.get("SHARDEVAL_REPO", PATCH))
import numpy as np
from unittest.mock import MagicMock

def mod(name, **attrs):
    m = types.ModuleType(name); m.__path__ = []
    for k, v in attrs.items(): setattr(m, k, v)
    sys.modules[name] = m
    return m

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m

BROADCAST_LOG = []
class ParticipatingNode:
    def __init__(self, id, env, location, params): pass
def get_transaction_delay(mu, sigma):          # depends on the GLOBAL numpy RNG, like the real helper
    d = np.random.normal(mu, sigma)
    while d <= 0: d = np.random.normal(mu, sigma)
    return d
helper = mod("utils.helper"); helper.get_transaction_delay = get_transaction_delay
helper.__getattr__ = lambda name: MagicMock()
mod("utils")
mod("matplotlib"); mod("matplotlib.font_manager", json_dump=None)
mod("interruptingcow", timeout=None)
mod("nodes"); mod("nodes.participating_node", ParticipatingNode=ParticipatingNode)
mod("network"); mod("network.broadcast", broadcast=lambda *a, **k: BROADCAST_LOG.append(a[1]))
for n in ["mini_block:MiniBlock", "tx_block:TxBlock", "cross_shard_block:CrossShardBlock", "block:Block", "pipe:Pipe"]:
    f, c = n.split(":"); mod(f"network.{f}", **{c: MagicMock()})
mod("network.consensus"); mod("network.consensus.consensus", Consensus=MagicMock())
mod("factory"); load("factory.transaction", os.path.join(REPO, "factory", "transaction.py"))
mod("factory.transaction_pool", TransactionPool=MagicMock())
fn = load("nodes.full_node", os.path.join(PATCH, "nodes", "full_node.py"))

class Env:
    def __init__(self): self.now = 0.0
    def timeout(self, d): return d

def make_params(P=None, cross=0.9, shards=5, seed=11):
    return dict(seed=seed, transaction_mu=2.0, transaction_sigma=1.0, tx_value_low=1, tx_value_high=100,
                reward_percentage=0.01, cross_shard_tx_percentage=cross, num_shards=shards, verbose=0,
                traffic_matrix=P, generated_tx_count=0, generated_intra_shard_tx_count=0, generated_cross_shard_tx_count=0)

def make_node(nid, shard, env, params):
    nd = object.__new__(fn.FullNode)
    nd.id, nd.params, nd.env, nd.shard_id = nid, params, env, shard
    nd.curr_shard_nodes, nd.next_hop_id, nd.node_type = {nid: nd}, -1, 3
    return nd

def run(nodes_spec, params, horizon, interfere, order_seed):
    """event-driven run of several node generators; `interfere` consumes the GLOBAL RNGs like voting/consensus would"""
    env = Env(); out = {nid: [] for nid, _ in nodes_spec}
    BROADCAST_LOG.clear()
    nodes = [make_node(nid, sh, env, params) for nid, sh in nodes_spec]
    gens, heap, cnt = {}, [], 0
    for nd in nodes:
        g = nd.generate_transactions(); gens[nd.id] = g
        heapq.heappush(heap, (next(g), cnt, nd.id)); cnt += 1
    rng = random.Random(order_seed)
    while heap:
        t, _, nid = heapq.heappop(heap)
        if t > horizon: break
        env.now = t
        before = len(BROADCAST_LOG)
        d = next(gens[nid])
        tx = BROADCAST_LOG[-1]; out[nid].append((tx.id, round(tx.creation_time, 9), tx.value, bool(tx.cross_shard_status), tx.target_shard_id))
        if interfere:                                   # different modes consume the global RNG differently
            for _ in range(rng.randint(0, 7)): np.random.random()
            for _ in range(rng.randint(0, 3)): random.random()
        heapq.heappush(heap, (t + d, cnt, nid)); cnt += 1
    return out

spec = [("FN1", 0), ("FN2", 1), ("FN3", 2), ("FN4", 0)]
n = 5
W = np.zeros((n, n)); r = 1
for i in range(n):
    for j in range(i + 1, n): W[i][j] = W[j][i] = 1.0 / r ** 1.5; r += 1
P = (W / W.sum(axis=1, keepdims=True)).tolist()

for label, Pm in [("uniform", None), ("skewed", P)]:
    np.random.seed(5); random.seed(5)
    a = run(spec, make_params(Pm), 400, interfere=False, order_seed=1)
    np.random.seed(5); random.seed(5)
    b = run(spec, make_params(Pm), 400, interfere=True, order_seed=2)
    same = all(a[k] == b[k] for k in a)
    total = sum(len(v) for v in a.values())
    print(f"[{label}] identical transaction streams with/without global-RNG interference: {same}  ({total} txs)")
    assert same and total > 200

# target distribution vs the traffic matrix, and own shard never targeted
np.random.seed(1); random.seed(1)
env = Env(); params = make_params(P, cross=1.0)
nd = make_node("FN9", 2, env, params); g = nd.generate_transactions(); next(g)
cnts = np.zeros(n); BROADCAST_LOG.clear()
for _ in range(20000):
    env.now += 1; next(g); cnts[BROADCAST_LOG[-1].target_shard_id] += 1
freq = cnts / cnts.sum()
print("target freq (shard 2):", np.round(freq, 3), " expected:", np.round(P[2], 3))
assert freq[2] == 0 and np.abs(freq - np.array(P[2])).max() < 0.012
np.random.seed(1); random.seed(1)
params = make_params(None, cross=1.0); nd = make_node("FN8", 3, Env(), params); g = nd.generate_transactions(); next(g)
cnts = np.zeros(n); BROADCAST_LOG.clear()
for _ in range(20000): next(g); cnts[BROADCAST_LOG[-1].target_shard_id] += 1
print("uniform targets (shard 3):", np.round(cnts / cnts.sum(), 3))
assert cnts[3] == 0 and np.abs(cnts / cnts.sum() - 0.25)[[0, 1, 2, 4]].max() < 0.012
ids = [t.id for t in BROADCAST_LOG]; assert len(set(ids)) == len(ids)
print("tx ids unique and deterministic, e.g.", ids[:2])
print("TEST full_node: PASS")

"""
MODIFIED transaction_pool.py for Sharon integration.

WHAT CHANGED vs. the original:
    - self.cross_shard_tx_queue (a single flat PriorityQueue) is replaced
      by self.cross_shard_tx_queues: a dict {target_shard_id: PriorityQueue}.
      This is the load-bearing change Sharon needs: to schedule which
      shards should merge, we must know the pending tx VOLUME PER TARGET
      SHARD *before* the round happens - not decide the target randomly
      at dequeue time, as the original preprocess_cross_shard_transactions
      did via get_cross_shard_random_node_id().

    - Transactions now carry a `target_shard_id` attribute, expected to be
      set at generation time (see full_node.py patch: generate_transactions
      must assign target_shard_id when creating a cross-shard tx, instead
      of leaving the destination undetermined).

    - get_transaction / pop_transaction / put_transaction for cross-shard
      txs now take an additional `target_shard_id` argument.

    - New method get_cross_shard_queue_lengths() returns a dict
      {target_shard_id: pending_count} for this node - this is exactly
      what a shard leader needs to compute one row of the W_{i,j} demand
      matrix if you build the load-aware (weighted) scheduling extension
      on top of vanilla Sharon later.

Everything else (locations, delay model, broadcast-on-accept, dedup via
prev_transactions) is unchanged from the original.
"""

import numpy as np

from network.broadcast import broadcast
from factory.transaction import Transaction
from utils.helper import get_transmission_delay
from utils.priority_queue import PriorityQueue


class TransactionPool:
    """
    Transaction pool for a full node
    """

    def __init__(self, env, id, neighbours_ids, nodes, params):
        self.env = env
        self.id = id
        self.neighbours_ids = neighbours_ids
        self.params = params
        self.nodes = nodes
        self.intra_shard_tx_queue = PriorityQueue()

        # CHANGED: one queue per target shard instead of a single flat queue.
        # Lazily created on first insert (see put_transaction) so this works
        # for any num_shards without pre-allocating params["num_shards"]
        # empty queues up front.
        self.cross_shard_tx_queues = {}

        self.prev_transactions = []
        self.intra_shard_tx = []
        self.cross_shard_tx = []

    def _get_or_create_cross_shard_queue(self, target_shard_id):
        if target_shard_id not in self.cross_shard_tx_queues:
            self.cross_shard_tx_queues[target_shard_id] = PriorityQueue()
        return self.cross_shard_tx_queues[target_shard_id]

    def get_transaction(self, transaction_count, tx_type, target_shard_id=None):
        """
        Return transaction_count number of Transactions.
        Returns top transactions based on tx reward.

        For tx_type == 'cross-shard', target_shard_id is now REQUIRED:
        it selects which per-target queue to read from.
        """
        if tx_type == 'intra-shard':
            return self.intra_shard_tx_queue.get(transaction_count)
        elif tx_type == 'cross-shard':
            if target_shard_id is None:
                raise RuntimeError(
                    "target_shard_id is required for cross-shard get_transaction() "
                    "under the Sharon-keyed transaction pool"
                )
            queue = self.cross_shard_tx_queues.get(target_shard_id)
            if queue is None:
                return []
            return queue.get(transaction_count)
        else:
            raise RuntimeError("Unknown transaction type specified")

    def pop_transaction(self, transaction_count, tx_type, target_shard_id=None):
        """
        Remove transactions from transaction pool.
        Called when transactions are added by a received block or a block is mined.
        """
        popped_transactions = None
        if tx_type == 'intra-shard':
            popped_transactions = self.intra_shard_tx_queue.pop(transaction_count)
        elif tx_type == 'cross-shard':
            if target_shard_id is None:
                raise RuntimeError(
                    "target_shard_id is required for cross-shard pop_transaction() "
                    "under the Sharon-keyed transaction pool"
                )
            queue = self.cross_shard_tx_queues.get(target_shard_id)
            popped_transactions = queue.pop(transaction_count) if queue else []
        else:
            raise RuntimeError("Unknown transaction type specified")

        self.prev_transactions.append(popped_transactions)
        return popped_transactions

    def put_transaction(self, transaction, source_location, tx_type):
        """
        Add received transaction to the transaction pool and broadcast further.

        For cross-shard transactions, `transaction.target_shard_id` MUST be
        set by the caller (see full_node.py's generate_transactions patch)
        before this is invoked - it determines which per-target queue the
        tx lands in.
        """
        if not tx_type == 'intra-shard' and not tx_type == 'cross-shard':
            raise RuntimeError("Unknown transaction type specified")

        dest_location = self.nodes[self.id].location
        delay = get_transmission_delay(source_location, dest_location)

        if tx_type == 'intra-shard':
            curr_queue = self.intra_shard_tx_queue
        else:
            if getattr(transaction, "target_shard_id", None) is None:
                raise RuntimeError(
                    f"Cross-shard transaction {transaction.id} has no target_shard_id set; "
                    "target shard must be assigned at generation time under Sharon"
                )
            curr_queue = self._get_or_create_cross_shard_queue(transaction.target_shard_id)

        yield self.env.timeout(delay)
        if (
            not curr_queue.is_present(transaction)
            and transaction not in self.prev_transactions
        ):
            curr_queue.insert(transaction)

            curr_node = self.nodes[self.id]
            neighbour_ids = [self.id if curr_node.next_hop_id == -1 else curr_node.next_hop_id]
            broadcast(self.env, transaction, "Tx", self.id, neighbour_ids, self.nodes, self.params)

            if self.params["verbose"]:
                print(
                    "%7.4f : %s accepted by %s"
                    % (self.env.now, transaction.id, self.id)
                )

    def get_cross_shard_queue_lengths(self):
        """
        Returns {target_shard_id: pending_tx_count} for this node's
        outgoing cross-shard queues.

        This is the per-node building block for Sharon's leader-level
        workload vector V_i = [|Queue(i->0)|, |Queue(i->1)|, ...] and,
        if you build the load-aware scheduling extension, for the
        symmetric demand matrix W_{i,j} = |Queue(i->j)| + |Queue(j->i)|.
        Vanilla Sharon (Algorithm 1-4) does not need this - it schedules
        obliviously - but it costs nothing to expose it now.
        """
        return {
            target_shard_id: queue.length()
            for target_shard_id, queue in self.cross_shard_tx_queues.items()
        }

    def get_cross_shard_pending_count(self, target_shard_id):
        queue = self.cross_shard_tx_queues.get(target_shard_id)
        return queue.length() if queue else 0

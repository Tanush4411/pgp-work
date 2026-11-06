from network.block import Block


class JointShardBlock(Block):
    """
    Sharon's replacement for CrossShardBlock.

    Where CrossShardBlock models ONE originating shard broadcasting to and
    collecting votes from EVERY other shard (Sharon Fig. 1(b)/(c): a
    special shard / special nodes coordinate cross-shard validation),
    JointShardBlock models exactly TWO shards merging for one round
    (Sharon Fig. 1(d)): shard_a and shard_b become a single joint shard,
    process only the transactions that cross between them, and every
    member of the joint shard votes once. There is no fan-out to shards
    outside this pairing and no Principal Committee relay required to
    combine sub-results, because the transaction is never split.

    This directly implements the "transaction isolation" property Sharon
    argues for in Section I: tx_a crossing (shard_a, shard_b) can never
    have its validation outcome entangled with tx_b crossing (shard_b,
    shard_c), because those are two different joint shards formed in
    (possibly) two different rounds.
    """

    def __init__(self, id, transactions_list, params, shard_a_id, shard_b_id,
                 shard_a_nodes, shard_b_nodes, round_number):
        super().__init__(id, transactions_list, params)
        self.shard_a_id = shard_a_id
        self.shard_b_id = shard_b_id
        self.round_number = round_number

        # Every node in BOTH shards votes on every tx in this joint block.
        # Flat dict (unlike CrossShardBlock's per-shard nested dict) because
        # there is only ever one "shard" involved here: the joint shard.
        self.votes_status = {}
        joint_nodes = list(shard_a_nodes) + list(shard_b_nodes)
        for tx in transactions_list:
            self.votes_status[tx.id] = {node_id: -1 for node_id in joint_nodes}
            # -1 = not yet voted, 0 = reject, 1 = accept

        self.visitor_count_post_voting = {}

    def get_joint_node_count(self):
        if not self.votes_status:
            return 0
        any_tx_id = next(iter(self.votes_status))
        return len(self.votes_status[any_tx_id])

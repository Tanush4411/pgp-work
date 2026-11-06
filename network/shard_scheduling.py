"""
Implements the Optimal Shard Scheduling algorithm from:

    Jiang, S., Cao, J., Tung, C.L., Wang, Y., Wang, S. (2024).
    "SHARON: Secure and Efficient Cross-shard Transaction Processing
    via Shard Rotation." IEEE INFOCOM 2024.

This module is intentionally self-contained (no dependency on the rest
of ShardEval) so it can be unit-tested in isolation before being wired
into network/network.py.

Terminology (matches the paper):
    n       - number of shards, indexed 0..n-1
    round   - one "meeting slot"; a shard merges with at most one other
              shard per round
    schedule - a list of rounds; each round is a list of (i, j) pairs
               that merge in that round

Algo 1 (odd n >= 3):  n rounds, one shard idle per round
Algo 2 (n == 2):      1 round
Algo 3 (even n >= 4): n-1 rounds, built from Algo 1 on n-1 shards
Algo 4:               dispatcher over the three cases above
"""

from typing import List, Tuple

Pair = Tuple[int, int]
Round = List[Pair]
Schedule = List[Round]


def algo1_odd(n: int) -> Schedule:
    """Optimal schedule for an odd number of shards (n >= 3). Runs in n rounds."""
    if n < 3 or n % 2 == 0:
        raise ValueError("algo1_odd requires odd n >= 3")

    schedule: Schedule = [[] for _ in range(n)]

    for r in range(1, n + 1):
        met = [False] * n
        for i in range(n):
            if met[i]:
                continue
            j = (r - 1 - i) % n
            if i == j:
                # Si tries to meet itself this round -> sits out
                continue
            met[i] = True
            met[j] = True
            schedule[r - 1].append((i, j))

    return schedule


def algo2_two_shards() -> Schedule:
    """Trivial case: n == 2. A single round where S0 meets S1."""
    return [[(0, 1)]]


def algo3_even(n: int) -> Schedule:
    """Optimal schedule for an even number of shards (n >= 4, n != 2).

    Built on top of algo1_odd(n-1): whichever shard sits idle in a given
    round of the (n-1)-shard schedule is paired with the new shard
    (index n-1) for that round. Runs in n-1 rounds.
    """
    if n < 4 or n % 2 != 0:
        raise ValueError("algo3_even requires even n >= 4")

    base = algo1_odd(n - 1)  # n-1 rounds over shards 0..n-2
    schedule: Schedule = []

    for round_pairs in base:
        met = [False] * (n - 1)
        for (i, j) in round_pairs:
            met[i] = True
            met[j] = True

        idle = [i for i in range(n - 1) if not met[i]]
        assert len(idle) == 1, "algo1_odd should leave exactly one shard idle per round"

        new_round = list(round_pairs) + [(idle[0], n - 1)]
        schedule.append(new_round)

    return schedule


def algo4_optimal_schedule(n: int) -> Schedule:
    """Dispatcher: returns the optimal meeting schedule for n shards.

    n == 2          -> algo2_two_shards()
    n odd,  n >= 3   -> algo1_odd(n)
    n even, n >= 4   -> algo3_even(n)
    """
    if n < 2:
        raise ValueError("Shard scheduling requires at least 2 shards")
    if n == 2:
        return algo2_two_shards()
    if n % 2 == 1:
        return algo1_odd(n)
    return algo3_even(n)


def validate_schedule(n: int, schedule: Schedule) -> None:
    """Sanity-checks a schedule against Sharon's correctness properties:
       1) validity   - no shard meets itself or two shards in the same round
       2) completeness - every pair of shards meets in exactly one round
       3) optimality  - number of rounds matches the paper's proven bound
    Raises AssertionError with a descriptive message on failure.
    """
    seen_pairs = set()

    for round_idx, round_pairs in enumerate(schedule):
        seen_this_round = set()
        for (i, j) in round_pairs:
            assert i != j, f"Round {round_idx}: shard {i} matched with itself"
            assert 0 <= i < n and 0 <= j < n, f"Round {round_idx}: index out of range ({i},{j})"
            assert i not in seen_this_round, f"Round {round_idx}: shard {i} appears twice"
            assert j not in seen_this_round, f"Round {round_idx}: shard {j} appears twice"
            seen_this_round.add(i)
            seen_this_round.add(j)

            pair = tuple(sorted((i, j)))
            assert pair not in seen_pairs, f"Pair {pair} met more than once (round {round_idx})"
            seen_pairs.add(pair)

    expected_pairs = n * (n - 1) // 2
    assert len(seen_pairs) == expected_pairs, (
        f"Only {len(seen_pairs)}/{expected_pairs} shard pairs met — schedule incomplete"
    )

    if n == 2:
        expected_rounds = 1
    elif n % 2 == 1:
        expected_rounds = n
    else:
        expected_rounds = n - 1
    assert len(schedule) == expected_rounds, (
        f"Expected {expected_rounds} rounds for n={n}, got {len(schedule)}"
    )


if __name__ == "__main__":
    # Quick self-test across a range of shard counts, mirroring Fig. 3 of the paper.
    for n in range(2, 21):
        schedule = algo4_optimal_schedule(n)
        validate_schedule(n, schedule)
        print(f"n={n:2d}: {len(schedule)} rounds -> OK  ({schedule})")
    print("\nAll shard-scheduling self-tests passed.")

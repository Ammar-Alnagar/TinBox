# Solution: Module 08 - Prefix Caching & Automatic Prefix Reuse

## Exercise 1: Unit Test Exercise
See `tests/test_module_08.rs` for implementation.
- Inserts prefix blocks into `PrefixCache` and queries matching prefixes.
- Verifies exact token match counts and physical block retrieval.

## Exercise 2: Failure-Case Exercise (Cache Misses & Disjoint Prefixes)
- Queries the prefix cache with a sequence having completely disjoint initial tokens.
- Verifies that zero tokens are matched, an empty block list is returned, and no spurious blocks are assigned.

## Exercise 3: Measurement & Reasoning Exercise
- Question: In a multi-turn chat application, the system prompt contains 512 tokens. A user asks 10 questions sequentially in the same conversation.
  1. Without prefix caching, how many total prompt tokens are processed across the 10 turns?
  2. With prefix caching, how many times is the 512-token system prompt recomputed?
  3. How many total prefill FLOPs/tokens are saved on the system prompt across the 10 turns?
- Calculation:
  - Without prefix caching:
    Each turn recomputes the 512-token prompt: $10 \times 512 = 5,120 \text{ tokens}$.
  - With prefix caching:
    The prompt is computed once in turn 1 and reused in turns 2 through 10. Recomputations = 0.
  - Saved prompt computations:
    $9 \times 512 = 4,608 \text{ tokens saved}$.
- Answer: **5,120 tokens processed without cache**, **0 recomputations with cache**, and **4,608 tokens saved**.

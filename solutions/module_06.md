# Solution: Module 06 - Attention Kernels & Paged Attention

## Exercise 1: Unit Test Exercise
See `tests/test_module_06.rs` for implementation.
- Evaluates `PagedAttentionRef::forward_decode` on non-contiguous blocks (e.g. Block #1 and Block #3).
- Verifies output shape is `[num_heads, head_dim]`.
- Asserts that attention probabilities across historical tokens sum to 1.0.

## Exercise 2: Failure-Case Exercise (Block-Table Length Edge Case)
- Tests when `seq_len` exceeds the capacity represented in `block_table` (e.g. `seq_len = 20` but block table has only 1 block of 16 tokens).
- Slices only up to the available blocks without panicking, and verifies bounds handling.

## Exercise 3: Measurement & Reasoning Exercise
- Question: Consider a sequence of length $N = 2048$ with head dimension $D = 128$.
  1. What is the size in elements of the attention matrix $Q K^T$ if fully materialized?
  2. What is the size in elements stored at any moment by online softmax tiled with block size $B = 16$?
  3. How many times less memory is required during attention score accumulation?
- Calculation:
  - Full attention matrix elements:
    $$N \times N = 2048 \times 2048 = 4,194,304 \text{ elements}$$
    For 1 query token (decode step):
    $$1 \times N = 2,048 \text{ elements}$$
  - Tiled online softmax elements stored at one step:
    $$1 \times B = 1 \times 16 = 16 \text{ elements}$$
  - Ratio of memory saved during decode score computation:
    $$\frac{2048}{16} = 128\times$$
- Answer: **2,048 elements** for decode step, **16 elements** for online softmax tile, and **128x less memory**.

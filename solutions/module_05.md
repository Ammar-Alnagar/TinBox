# Solution: Module 05 - Model Execution Backend

## Exercise 1: Unit Test Exercise
See `tests/test_module_05.rs` for implementation.
- Creates a `ForwardBatch` with 2 sequences in ragged 1D representation.
- Runs the `CandleBackend::forward` pass and verifies output logits shape is `[2, vocab_size]`.

## Exercise 2: Failure-Case Exercise (Empty Batch Execution)
- Attempting to run a forward pass with an empty `ForwardBatch` (`input_tokens = vec![]`, `positions = vec![]`) returns `Err(InferError::Backend)`.
- Verifies that the engine does not pass invalid 0-dimension tensors to the computation framework.

## Exercise 3: Measurement & Reasoning Exercise
- Question: Consider a continuous batch of 4 sequences with lengths 10, 25, 4, and 61 tokens.
  1. What is the total length of the 1D ragged token array?
  2. If traditional static batching were used with padding to the maximum sequence length (61 tokens), how many total tokens would be processed, and what percentage of compute is wasted on padding?
- Calculation:
  - Ragged total tokens:
    $$10 + 25 + 4 + 61 = 100 \text{ tokens}$$
  - Padded batch total tokens:
    $$4 \times 61 = 244 \text{ tokens}$$
  - Wasted padding tokens:
    $$244 - 100 = 144 \text{ tokens}$$
  - Waste percentage:
    $$\frac{144}{244} \times 100\% \approx 59.02\%$$
- Answer: **100 tokens** in ragged batching, **244 tokens** in padded batching, and **59.02% wasted compute**.

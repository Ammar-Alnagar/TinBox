# Solution: Module 09 - Sampling & Output Processing

## Exercise 1: Unit Test Exercise
See `tests/test_module_09.rs` for implementation.
- Verifies deterministic greedy sampling (`temperature = 0.0`) selects the argmax logit.
- Verifies multiplicative repetition penalty reduces repeated token logits.

## Exercise 2: Failure-Case Exercise (Empty Logits Slice)
- Calling `Sampler::sample_token(&[], ...)` with an empty slice returns `Err(InferError::InvalidRequest)`.
- Verifies that zero-length vocabulary slices are rejected safely.

## Exercise 3: Measurement & Reasoning Exercise
- Question: Suppose unnormalized logits are $z = [2.0, 4.0]$.
  1. What are the Softmax probabilities at temperature $T = 1.0$?
  2. What are the Softmax probabilities at temperature $T = 0.5$?
- Calculation:
  - At $T = 1.0$:
    $$\Delta z = 2.0 - 4.0 = -2.0$$
    $$e^{2.0 - 4.0} = e^{-2} \approx 0.1353, \quad e^{4.0 - 4.0} = 1.0$$
    $$\sum = 1.1353$$
    $$P(z_0) = \frac{0.1353}{1.1353} \approx 0.1192 \quad (11.92\%)$$
    $$P(z_1) = \frac{1.0}{1.1353} \approx 0.8808 \quad (88.08\%)$$
  - At $T = 0.5$:
    Scaled logits: $z' = [4.0, 8.0]$.
    $$e^{4.0 - 8.0} = e^{-4} \approx 0.0183, \quad e^{8.0 - 8.0} = 1.0$$
    $$\sum = 1.0183$$
    $$P(z_0) = \frac{0.0183}{1.0183} \approx 0.0180 \quad (1.80\%)$$
    $$P(z_1) = \frac{1.0}{1.0183} \approx 0.9820 \quad (98.20\%)$$
- Answer: At $T=1.0$: **[0.1192, 0.8808]**; at $T=0.5$: **[0.0180, 0.9820]**. Lower temperature sharpens probabilities toward the argmax.

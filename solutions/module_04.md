# Solution: Module 04 - Model Loading & FP8 Representation

## Exercise 1: Unit Test Exercise
See `tests/test_module_04.rs` for implementation.
- Verifies exact bitwise FP8 E4M3 numerics: `0b00111000` $\to 1.0$, `0b10111000` $\to -1.0$, and subnormals `0b00000001` $\to 0.001953125$.

## Exercise 2: Failure-Case Exercise (E4M3 NaN Representation)
- Tests that byte `0b01111111` (all exponent and mantissa bits 1) correctly decodes to IEEE 754 `NaN`.
- Verifies that calling `.is_nan()` on the decoded float returns `true`.

## Exercise 3: Measurement & Reasoning Exercise
- Question: A 7-billion parameter language model is loaded in 16-bit float (BF16 / FP16) requiring 14 GB of memory.
  1. What is the memory footprint when weights are quantized to FP8 (1 byte per parameter)?
  2. What is the percentage reduction in memory footprint?
- Calculation:
  $$\text{FP8 Memory} = 7 \times 10^9 \text{ bytes} = 7.0 \text{ GB}$$
  $$\text{Reduction} = \frac{14 \text{ GB} - 7 \text{ GB}}{14 \text{ GB}} \times 100\% = 50.0\%$$
- Answer: **7.0 GB** and **50.0% reduction**.

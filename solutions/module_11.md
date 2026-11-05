# Solution: Module 11 - Integration, Testing & End-to-End Polish

## Exercise 1: Unit Test Exercise
See `tests/test_module_11.rs` for implementation.
- Instantiates full `EngineRuntime` with synthetic weights and continuous scheduler.
- Spawns background worker task and submits request over unbounded channel.
- Verifies streamed token generation until `is_finished == true`.

## Exercise 2: Failure-Case Exercise (Channel Overflow / Early Drop Shutdown)
- Drops the request sender channel while `run_loop` is executing.
- Verifies the engine terminates cleanly without leaking threads or panicking.

## Exercise 3: Measurement & Reasoning Exercise
- Question: An end-to-end inference benchmark on a synthetic model yields:
  - TTFT = 25 ms for a 128-token prompt.
  - TPOT = 5 ms per token during auto-regressive decode.
  1. What is the total latency to generate 64 tokens?
  2. What is the effective generation throughput in tokens per second for this sequence?
- Calculation:
  $$\text{Total Latency} = \text{TTFT} + (\text{Output Tokens} \times \text{TPOT})$$
  $$\text{Total Latency} = 25\text{ ms} + (64 \times 5\text{ ms}) = 25 + 320 = 345\text{ ms} = 0.345\text{ s}$$
  $$\text{Throughput} = \frac{64 \text{ tokens}}{0.345 \text{ s}} \approx 185.51 \text{ tokens/s}$$
- Answer: **345 ms total latency** and **185.51 tokens/second**.

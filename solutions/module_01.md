# Solution: Module 01 - Core Types & Request Lifecycle

## Exercise 1: Unit Test Exercise
See `tests/test_module_01.rs` for implementation.
- Creates a `Sequence` with prompt tokens `[1, 2, 3]` and `max_new_tokens = 3`.
- Tests that appending 3 tokens automatically sets `status = SequenceStatus::Finished(FinishReason::Length)`.
- Tests that adding an explicit stop token triggers `FinishReason::Stop`.

## Exercise 2: Failure-Case Exercise (Empty Prompt & Immediate Client Disconnect)
- An empty sequence (`prompt_tokens.is_empty()`) has `len() == 0` and `is_empty() == true`.
- When an HTTP client disconnects prematurely, the `mpsc::Receiver` is dropped. Calling `emit_token` fails silently on the unbounded sender (`let _ = sender.send(output);`) without causing the runtime loop to panic.

## Exercise 3: Measurement & Reasoning Exercise
- Question: Suppose a sequence has 256 prompt tokens and generates 64 output tokens. If each token requires 48 KB of KV cache per token, compute the total KV cache memory footprint of this sequence at completion.
- Calculation:
  $$\text{Total Tokens} = 256 + 64 = 320 \text{ tokens}$$
  $$\text{Memory} = 320 \times 48 \text{ KB} = 15,360 \text{ KB} = 15.0 \text{ MB}$$
- Answer: **15.0 MB** (or 15,360 KB).

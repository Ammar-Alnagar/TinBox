# Solution: Module 03 - KV Cache & Paged Memory Layout

## Exercise 1: Unit Test Exercise
See `tests/test_module_03.rs` for implementation.
- Tests logical-to-physical block table address translation.
- Tests sequence forking and Copy-on-Write (COW) when appending a token to a shared block.

## Exercise 2: Failure-Case Exercise (Block Size Edge Cases & Empty Prompts)
- Submitting an empty prompt to `allocate_for_prompt` returns `Err(InferError::InvalidRequest)`.
- Requesting translation of a token index exceeding the allocated block table bounds returns `Err(InferError::Scheduler)`.

## Exercise 3: Measurement & Reasoning Exercise
- Question: Suppose a model has 32 layers, 8 KV heads, head dimension 128, and FP16 precision (2 bytes). Tokens are grouped into blocks of `block_size = 16`.
  1. What is the size in bytes of a single physical KV block across all layers?
  2. For a sequence of 50 tokens, how many physical blocks are required, and how many bytes of internal fragmentation occur in the final block?
- Calculation:
  - Cache size per token across all layers:
    $$2 \times 32 \times 8 \times 128 \times 2 = 131,072 \text{ bytes} = 128 \text{ KB / token}$$
  - Size of 1 physical block (16 tokens):
    $$16 \times 128 \text{ KB} = 2,048 \text{ KB} = 2.0 \text{ MB}$$
  - For 50 tokens:
    $$\text{Blocks needed} = \lceil 50 / 16 \rceil = 4 \text{ blocks}$$
    Tokens in final block = $50 - 3 \times 16 = 2$ tokens.
    Empty token slots in final block = $16 - 2 = 14$ tokens.
    Internal fragmentation = $14 \times 128 \text{ KB} = 1,792 \text{ KB} = 1.75 \text{ MB}$.
- Answer: **2.0 MB** per block, **4 blocks** required, and **1.75 MB** internal fragmentation.

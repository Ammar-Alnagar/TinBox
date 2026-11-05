# Solution: Module 07 - Scheduler & Continuous Batching

## Exercise 1: Unit Test Exercise
See `tests/test_module_07.rs` for implementation.
- Tests scheduling of waiting sequences into running state when free blocks are available.
- Verifies distinction between prefill batch and decode batch.

## Exercise 2: Failure-Case Exercise (Preemption Under Memory Pressure)
- Configures an allocator with only 2 blocks.
- Adds 2 sequences, each needing 1 block for prompt.
- Appending a token to the running sequence when 0 blocks are free triggers preemption of the running sequence to `SequenceStatus::Preempted` and re-enqueues it to the waiting queue.

## Exercise 3: Measurement & Reasoning Exercise
- Question: A continuous batch scheduler processes a maximum batch size of 16 sequences. At tick $T_0$, 10 sequences are in decode phase and 4 sequences are in waiting queue with prompt lengths 32, 64, 16, and 128 tokens. Block size is 16.
  1. How many new blocks are required to admit all 4 waiting sequences?
  2. If only 8 free blocks remain in the allocator, which waiting sequences are scheduled (FIFO order), and how many free blocks remain?
- Calculation:
  - Blocks needed per waiting sequence:
    - Seq 1 (32 tokens): $32 / 16 = 2$ blocks.
    - Seq 2 (64 tokens): $64 / 16 = 4$ blocks.
    - Seq 3 (16 tokens): $16 / 16 = 1$ block.
    - Seq 4 (128 tokens): $128 / 16 = 8$ blocks.
    - Total needed: $2 + 4 + 1 + 8 = 15$ blocks.
  - With 8 free blocks:
    - Seq 1 scheduled: uses 2 blocks (6 remaining).
    - Seq 2 scheduled: uses 4 blocks (2 remaining).
    - Seq 3 scheduled: uses 1 block (1 remaining).
    - Seq 4 requires 8 blocks, but only 1 block is free -> cannot be scheduled.
    - Final free blocks remaining: $1$ block.
- Answer: **15 blocks needed**, **Sequences 1, 2, and 3 scheduled**, and **1 free block remaining**.

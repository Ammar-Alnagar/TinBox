# Solution: Module 02 - Memory Management Foundation

## Exercise 1: Unit Test Exercise
See `tests/test_module_02.rs` for implementation.
- Verifies block allocation, ref-counting, and free-list recycling.
- Verifies that retained blocks remain allocated until ref-count decrements to 0.

## Exercise 2: Failure-Case Exercise (Out of Memory)
- Requesting more blocks than available in `BlockAllocator::allocate_exact(count)` returns `Err(InferError::OutOfMemory)`.
- Verifies that upon failure, no partial blocks are leaked from the allocator.

## Exercise 3: Measurement & Reasoning Exercise
- Question: A system has 8 GB of VRAM dedicated to the KV block pool. Each block holds 16 tokens and takes 768 KB of memory.
  1. How many total blocks can be allocated?
  2. What is the total token capacity of the pool?
- Calculation:
  $$8 \text{ GB} = 8 \times 1024 \times 1024 \text{ KB} = 8,388,608 \text{ KB}$$
  $$\text{Total Blocks} = \left\lfloor \frac{8,388,608}{768} \right\rfloor = 10,922 \text{ blocks}$$
  $$\text{Total Tokens} = 10,922 \times 16 = 174,752 \text{ tokens}$$
- Answer: **10,922 blocks** and **174,752 tokens**.

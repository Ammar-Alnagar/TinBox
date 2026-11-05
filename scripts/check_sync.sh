#!/usr/bin/env bash
set -euo pipefail

# check_sync.sh: extracts Rust blocks from MODULE files and checks for synchronization with src/
python3 - << 'PYEOF'
import sys
import re
import difflib

# Mapping of (module_file, block_index) -> source_file
MAPPINGS = [
    ("MODULE_00_PROJECT_SCAFFOLD.md", 0, "src/error.rs"),
    ("MODULE_00_PROJECT_SCAFFOLD.md", 1, "src/config.rs"),
    ("MODULE_00_PROJECT_SCAFFOLD.md", 2, "src/main.rs"),
    ("MODULE_01_CORE_TYPES.md", 0, "src/types.rs"),
    ("MODULE_02_MEMORY_MANAGEMENT.md", 0, "src/memory/block_allocator.rs"),
    ("MODULE_03_KV_CACHE.md", 0, "src/memory/kv_cache.rs"),
    ("MODULE_04_MODEL_LOADING.md", 0, "src/model/config.rs"),
    ("MODULE_04_MODEL_LOADING.md", 1, "src/model/loader.rs"),
    ("MODULE_05_MODEL_EXECUTION.md", 0, "src/engine/backend.rs"),
    ("MODULE_06_ATTENTION_KERNELS.md", 1, "src/engine/paged_attention.rs"),
    ("MODULE_07_SCHEDULER.md", 0, "src/runtime/scheduler.rs"),
    ("MODULE_08_PREFIX_CACHING.md", 0, "src/memory/prefix_cache.rs"),
    ("MODULE_09_SAMPLING.md", 0, "src/engine/sampler.rs"),
    ("MODULE_10_SERVING_LAYER.md", 0, "src/serving/server.rs"),
    ("MODULE_11_INTEGRATION_TESTING.md", 0, "src/runtime/engine.rs"),
]

has_diff = False
for mod_file, b_idx, src_file in MAPPINGS:
    with open(mod_file, "r") as f:
        mod_content = f.read()
    blocks = re.findall(r"```rust\n(.*?)```", mod_content, re.DOTALL)
    if b_idx >= len(blocks):
        print(f"ERROR: {mod_file} has only {len(blocks)} blocks, requested index {b_idx}", file=sys.stderr)
        has_diff = True
        continue
    mod_block = blocks[b_idx].strip()
    with open(src_file, "r") as f:
        src_code = f.read().strip()

    if mod_block != src_code:
        print(f"DIFF DETECTED: {mod_file} (block {b_idx}) != {src_file}")
        diff = list(difflib.unified_diff(
            mod_block.splitlines(),
            src_code.splitlines(),
            fromfile=f"{mod_file}[{b_idx}]",
            tofile=src_file,
            lineterm=""
        ))
        print("\n".join(diff[:25]))
        print("-" * 50)
        has_diff = True
    else:
        print(f"OK: {mod_file} (block {b_idx}) == {src_file}")

if has_diff:
    print("\nCheck sync found differences!", file=sys.stderr)
    sys.exit(1)
else:
    print("\nAll mapped Rust blocks are in sync with src/!")
PYEOF

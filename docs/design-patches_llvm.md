#### 3. `_loop_ub_as_i64` is a fragile heuristic (L100-115)

```python
def _loop_ub_as_i64(index: SSAValue) -> SSAValue | None:
```

This function reverse-engineers the loop upper bound by tracing: `index -> unrealized_cast -> block_arg -> find ICmpOp in header block -> extract the other operand`. It assumes:
- The index is a loop IV (block arg at position 0).
- The header block has exactly one `ICmpOp` that compares the IV against the upper bound.
- The upper bound is `i64`.

If the loop structure changes (e.g., a different comparison, or the IV isn't at position 0, or there are multiple comparisons), this silently returns `None` and the caller asserts. The function has no way to report *why* it failed.

**Proposed change:** At minimum, add a docstring explaining what IR shape it expects and when it returns `None`. Ideally, carry the upper bound through metadata or a side table rather than reverse-engineering it from the CFG.

## Issues

### Severity: High

#### 2. `iconst` lambda duplicated (L122, L186)

```python
# in _get_target_ptr:
iconst = lambda n: ins(llvm.ConstantOp(IntegerAttr(n, i64), i64)).result

# in ConvertSubviewPattern:
iconst = lambda n: ins(llvm.ConstantOp(IntegerAttr(n, i64), i64)).result
```

Identical lambda. Part of the duplication in issue #1.

**Proposed change:** Make `iconst` a module-level helper or part of the extracted `_offset_ptr`.

### Severity: Medium

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

### Severity: Low

#### 5. `RewriteMemRefTypes.recursive = True` is never changed (L254)

```python
@dataclass
class RewriteMemRefTypes(TypeConversionPattern):
    recursive: bool = True
```

The `recursive` field is always `True` (set by the dataclass default). It's never assigned any other value. If `TypeConversionPattern` requires it, it should be a class constant. If it doesn't, it's dead code.

**Proposed change:** Check if `TypeConversionPattern` reads `self.recursive`. If yes, set it as a class variable: `recursive: ClassVar[bool] = True`. If no, remove it.

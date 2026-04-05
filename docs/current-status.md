# Callbot Engine Hush — Current Status (2026-04-06)

## Scheduler Inline Dispatch + BaseOp Hot Path Optimization

### Problem
The Hush scheduler dispatches ALL ops via `asyncio.create_task()` + queue, adding ~96ms overhead for a 693-chunk streaming pipeline (x2 latency vs direct BaseOp calls).

### What was done

#### Phase 1: Scheduler inline dispatch (hush-icore)
1. **Unified `self.bound`** — merged `self.executor`, `self._dispatch_type` into single `self.bound`:
   - `"sync"` — inline dispatch, no asyncio task, no queue (fastest)
   - `"io"` — `asyncio.create_task()` for async I/O ops
   - `"cpu"` — `asyncio.to_thread()` for heavy compute (C extensions release GIL)
   - `None` — auto-detect: sync fn → `"sync"`, async fn → `"io"`

2. **Scheduler inline path** (`task_scheduler.py`) — sync ops bypass `asyncio.create_task()` + queue entirely via `_drain_inline()`.

3. **GraphOp auto-detect** — all children sync → graph becomes sync. Override with `@graph(bound="io")`.

4. **`@graph(bound=...)` decorator** — `@graph` now accepts kwargs.

5. **Backward compat** — `@op(executor="thread")` maps to `bound="cpu"`.

6. **`_exec_core()` optimized** — uses precomputed `self.bound` instead of 4x `inspect.*` calls per invocation.

#### Phase 2: BaseOp.run() hot path (hush-icore)

Profiled with no-op functions (693 chunks × 3 ops = 2081 invocations) to isolate pure framework cost.

7. **`datetime.now()` conditional** — only called when `state.tracing=True` (tracer attached). Saves 2 datetime calls per op.

8. **`_log()` conditional** — only called when tracing. Eliminates per-op format_event + logger check.

9. **`get_inputs()` cached indices** — on first call, caches `(var_name, cell_index, fallback)` tuples. Subsequent calls skip `schema.get_index()` dict lookup per input param. `state[]` gets dropped from 2774 → 6.

10. **`_store_metrics()` cached indices** — caches `(st_idx, et_idx, dur_idx)`. Writes directly to `cells[idx][ctx]` bypassing `_unpack_key` + `get_index`. State sets dropped from 11,095 → 2,772.

11. **Error state cached index** — same pattern for `state[op, "error", ctx]` writes.

12. **`state.tracing` flag** — `MemoryState.tracing` defaults to `True` (backward compat), engine sets `False` when no tracer attached.

13. **`iter_executed()` uses `duration_ms`** — always present (even without tracer), instead of `start_time` which is `None` when not tracing.

#### callbot-engine-hush changes
1. `speech/vad_detector_v2.py` — `speech_segmenter` from `async def` to `def` (enables inline dispatch)
2. Benchmark/profiling scripts in `scripts/`

### Test results
- **727 tests pass** in hush-icore (19 new + 708 existing)
- **10/10 exact match** between VAD v1 and v2 output

### Benchmark results

#### Pure framework overhead (693 chunks × 3 no-op ops)
| Stage | Before | After | Saved |
|-------|--------|-------|-------|
| `state[] set` | 6.2ms (11,095 calls) | 2.5ms (2,772 calls) | -60% |
| `_store_metrics()` | 5.7ms | 1.8ms | -68% |
| `get_inputs()` | 5.3ms | 3.0ms | -43% |
| `state[] get` | 3.3ms (2,774 calls) | 0.0ms (6 calls) | -99% |
| `_log()` | 1.1ms | 0.0ms | -100% |
| **Total run** | **42ms** | **31ms** | **-26%** |

#### Real VAD pipeline (693 chunks, 10 WAVs concatenated)
| Version | Time | Overhead |
|---------|------|----------|
| V1 (direct BaseOp) | ~100-120ms | baseline |
| V2 (@graph, before) | ~196ms | +96ms |
| V2 (@graph, after) | ~183ms | +82ms |

### Remaining overhead breakdown (31ms framework, no-op)
| Component | Time | % | Notes |
|-----------|------|---|-------|
| async gen protocol + scheduler | 18.5ms | 61% | Python async machinery floor |
| `store_result()` | 4.7ms | 15% | Dynamic output keys, hard to cache |
| `get_inputs()` | 3.0ms | 10% | Already cached, residual loop cost |
| `state[] set` (from store_result) | 2.5ms | 8% | Output writes |
| `_store_metrics()` | 1.8ms | 6% | Cached, 3 cell writes |

The 18.5ms async generator protocol cost is the floor — Python's `async for ... yield` machinery per chunk.

### Files changed

#### Hush-ai (hush-icore, `dev` branch)
- `hush/core/ops/base.py` — unified bound, cached get_inputs/metrics/error indices, conditional datetime/log
- `hush/core/ops/graph/task_scheduler.py` — inline dispatch + `_drain_inline()`
- `hush/core/ops/graph/graph_op.py` — GraphOp bound auto-detect from children
- `hush/core/ops/graph/_decorators.py` — `@graph(bound=...)` support
- `hush/core/ops/transform/func_op.py` — `executor` → `bound` mapping
- `hush/core/ops/_shortcuts.py` — removed `executor` from `_BASE_INIT_KEYS`
- `hush/core/states/state.py` — `tracing` flag, `iter_executed` uses `duration_ms`
- `hush/core/engine.py` — sets `state.tracing` from tracer presence
- `tests/ops/graph/test_scheduler_bound.py` — 19 new tests
- `tests/ops/transform/test_executor.py` — updated to use `bound="cpu"`
- `tests/states/test_state.py` — updated iter_executed tests
- `tests/tracing/test_local_tracer.py` — uses dummy tracer for tracing tests

#### callbot-engine-hush
- `speech/vad_detector_v2.py` — `speech_segmenter` sync for inline dispatch
- `scripts/compare_vad_v1_v2.py` — VAD v1 vs v2 benchmark with concat test
- `scripts/profile_framework_only.py` — pure framework overhead profiling
- `scripts/profile_baseop_run.py` — BaseOp.run() breakdown profiling

# Callbot Engine Hush — Current Status (2026-04-06)

## Scheduler Inline Dispatch Optimization (in progress)

### Problem
The Hush scheduler dispatches ALL ops via `asyncio.create_task()` + queue, adding ~96ms overhead for a 693-chunk streaming pipeline (x2 latency vs direct BaseOp calls).

### What was done

#### hush-icore changes (on Hush-ai `dev` branch)
1. **Unified `self.bound`** — merged `self.executor`, `self._dispatch_type` into a single `self.bound` attribute:
   - `"sync"` — inline dispatch, no asyncio task, no queue (fastest)
   - `"io"` — `asyncio.create_task()` for async I/O ops
   - `"cpu"` — `asyncio.to_thread()` for heavy compute (C extensions release GIL)
   - `None` — auto-detect: sync fn → `"sync"`, async fn → `"io"`

2. **Scheduler inline path** (`task_scheduler.py`) — sync ops bypass `asyncio.create_task()` + queue entirely. Uses `_drain_inline()` to process Frame/EOF events directly in the call stack.

3. **GraphOp auto-detect** — if all children are `"sync"`, the graph itself becomes `"sync"` (inline). User can override with `@graph(bound="io")`.

4. **`@graph(bound=...)` decorator** — `@graph` now accepts kwargs like `@op` does.

5. **Backward compat** — `@op(executor="thread")` still works, maps to `bound="cpu"`.

6. **`_exec_core()` optimized** — uses precomputed `self.bound` instead of 4x `inspect.*` calls per invocation.

#### callbot-engine-hush changes
1. `speech/vad_detector_v2.py` — changed `speech_segmenter` from `async def` to `def` (no actual async work, enables inline dispatch)
2. New benchmark/comparison scripts in `scripts/`

### Test results
- **727 tests pass** in hush-icore (19 new scheduler bound tests + 708 existing)
- **10/10 exact match** between VAD v1 and v2 output

### Benchmark results (693 chunks, all 10 WAVs concatenated)
| Version | Time | Per-chunk |
|---------|------|-----------|
| V1 (direct BaseOp) | 100ms | 0.14ms |
| V2 (@graph inline) | 196ms | 0.28ms |
| Overhead | +96ms | +0.14ms/chunk |

### Remaining work
The scheduler inline dispatch eliminated `asyncio.create_task()` + queue overhead, but **`BaseOp.run()` per-op wrapping** still costs ~0.14ms/chunk. Root causes to profile:
- `datetime.now(timezone.utc)` — 2 calls per op (start_time, end_time)
- `_store_metrics()` — 3 state writes per op (start_time, end_time, duration_ms)
- `_log()` — conditional but still checked per op
- `get_inputs()` — iterates all input params with state lookups
- `store_result()` — iterates all output keys with state writes
- `state[op, var, ctx]` access — dict lookup chain per read/write

Next step: instrument `BaseOp.run()` to measure each part, then optimize the hot path (e.g., skip metrics when no tracer, batch state writes, etc.)

### Files changed

#### Hush-ai (hush-icore)
- `hush/core/ops/base.py` — unified `bound`, removed `executor`/`_dispatch_type`
- `hush/core/ops/graph/task_scheduler.py` — inline dispatch + `_drain_inline()`
- `hush/core/ops/graph/graph_op.py` — GraphOp bound auto-detect from children
- `hush/core/ops/graph/_decorators.py` — `@graph(bound=...)` support
- `hush/core/ops/transform/func_op.py` — `executor` → `bound` mapping
- `hush/core/ops/_shortcuts.py` — removed `executor` from `_BASE_INIT_KEYS`
- `tests/ops/graph/test_scheduler_bound.py` — 19 new tests
- `tests/ops/transform/test_executor.py` — updated to use `bound="cpu"`

#### callbot-engine-hush
- `speech/vad_detector_v2.py` — `speech_segmenter` sync for inline dispatch
- `scripts/compare_vad_v1_v2.py` — VAD v1 vs v2 benchmark
- `scripts/profile_vad_v2.py` — profiling script
- `scripts/test_vad_v2_shared_time.py` — shared state test

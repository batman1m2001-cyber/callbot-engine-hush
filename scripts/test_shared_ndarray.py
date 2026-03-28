"""Debug: shared var ndarray push ref — does shape survive?"""

import asyncio
import numpy as np

from hush.core import END, PARENT, START, GraphOp, op
from hush.core.states import StateSchema


@op
def source(n: int):
    for i in range(n):
        yield {"x": i}


@op
def update_arr(x: int, arr: np.ndarray) -> dict:
    """Read arr, modify, return new arr."""
    new_arr = arr.copy()
    new_arr[0, 0, 0] = float(x)
    print(f"  x={x}: input arr shape={arr.shape}, dtype={arr.dtype}")
    return {"new_arr": new_arr, "result": float(new_arr[0, 0, 0])}


with GraphOp(name="test") as g:
    PARENT.shared(arr=np.zeros((2, 1, 128), dtype=np.float32))

    s = source(n=PARENT["n"])
    u = update_arr(x=s["x"], arr=PARENT["arr"])
    u["new_arr"] >> PARENT["arr"]
    START >> s >> u >> END

g.build()
schema = StateSchema(g)
state = schema.create_state(inputs={"n": 3})
result = asyncio.run(g.run(state))

print(f"\nresult['result'] = {result['result']}")
print(f"final arr = {result['arr']}")
if hasattr(result['arr'], 'shape'):
    print(f"final arr shape = {result['arr'].shape}")
else:
    print(f"final arr type = {type(result['arr'])}")

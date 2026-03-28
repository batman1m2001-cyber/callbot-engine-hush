"""Test: nested @graph streaming → downstream receives per-item or batched?"""

import asyncio
from hush.core import Hush, graph, START, END, PARENT, op


@op
async def source(n: int):
    for i in range(n):
        yield {"x": i}
        await asyncio.sleep(0)


@graph
def inner(x):
    """Nested graph: receives x, yields x*2."""
    @op
    def double(x: int):
        return {"y": x * 2}

    d = double(x=x)
    START >> d >> END


@op
def consumer(y: int) -> dict:
    print(f"  consumer received: y={y} type={type(y).__name__}")
    return {"result": y}


@graph
def pipeline(n):
    s = source(n=n)
    i = inner(x=s["x"])
    c = consumer(y=i["y"])
    START >> s >> i >> c >> END


async def main():
    wf = pipeline(n=3)
    engine = Hush(wf)
    result = await engine.run(inputs={})
    print(f"\nresult['result'] = {result.get('result')}")


asyncio.run(main())

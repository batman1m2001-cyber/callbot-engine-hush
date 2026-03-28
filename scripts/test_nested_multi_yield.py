"""Test: nested @graph with inner generator that yields MULTIPLE items per call."""

import asyncio
from hush.core import Hush, graph, START, END, PARENT, op


@op
async def source(n: int):
    for i in range(n):
        yield {"x": i}
        await asyncio.sleep(0)


@graph
def inner(x):
    """Inner graph: yields MULTIPLE items from one input."""
    @op
    async def expand(x: int):
        # Yield 2 items per input
        yield {"y": x * 10}
        yield {"y": x * 10 + 1}

    e = expand(x=x)
    START >> e >> END


@op
def consumer(y) -> dict:
    print(f"  consumer: y={y} type={type(y).__name__}")
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
    print(f"\nresult = {result.get('result')}")


asyncio.run(main())

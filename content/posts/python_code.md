---
author: [""]
title: "Python in practice"
date: 2025-10-10
summary: "Learnings from reading the python code for semlib library"
description: ""
tags: ["python", "best-practices"]
ShowToc: false
ShowBreadCrumbs: true
---

I was reading the code for [semlib](https://github.com/anishathalye/semlib), a Python library that lets you process and analyze data using LLM-powered operations like `map`, `filter`, and `reduce`. The author, [Anish Athalye](https://anishathalye.com/semlib/), has also written a great post explaining the motivation behind it.

The code is well written and thought out in terms of abstractions, unit tests and of course documentation. I wanted to capture a few learnings that I found valuable when thinking about Python code design.

Let's start with something basic. Suppose we want to ask an LLM about the color of each fruit in a list:

```python
`map(["apple", "banana", "kiwi"], template="What color is {}? Reply in a single word.")`.
```

The output in this case might be `['Red.', 'Yellow.', 'Green.']`.

So how does Semlib make this `map` function work?

**Base class**: Create a base [`Base`](https://github.com/anishathalye/semlib/blob/master/src/semlib/_internal/base.py) class that handles LLM interaction. It uses LiteLLM under the hood that provides support for different LLM providers.

Three things I found particularly interesting here:

1. **Concurrency control** – The use of `asyncio.Semaphore` to rate-limit concurrent LLM API calls.
2. **Caching** - Cache keys are built from `(messages, return type and the model)`, hashed to create a unique identifier. [Caching](https://github.com/anishathalye/semlib/blob/master/src/semlib/cache.py) itself is implemented using an inheritance-based approach, supporting both in-memory or disk caching. The use of async [`Condition`](https://docs.python.org/3/library/asyncio-sync.html#condition) helps coordinate cache fills and lookup asynchronously, while sets are used to keep track of the pending requests.
3. **Type safety** - The use of generics and overloading in Python. Generics and `@overload` are used effectively to make prompts and responses type-safe, improving both auto-complete and static checks.

**Map**: [`Map`](https://github.com/anishathalye/semlib/blob/master/src/semlib/map.py) class is dervied from `Base` class implemented above where each item in the list is passed to LLM. 

Three interesting things that stood out: 

1. **Two interfaces** - You can call `Map` either via a class instance or a standalone function. The use of positional (`/`) and keyword (`*`) arguments is well thought out.
2. **Type clarity** - Overloading and generics combine to make the API intuitive and correctly typed for multiple combinations of input and output types.
3. **Async and sync symmetry** – Every async function has a sync counterpart, so users can pick the right abstraction without worrying about the implementation.

The library also implements other LLM-powered primitives such as `filter`, `reduce`, `sort`, `compare`, `apply`.

Finally, the test suite is equally well-designed. I particularly liked the approach taken to implement the unit tests for `Base` class in [`conftest.py`](https://github.com/anishathalye/semlib/blob/master/tests/conftest.py). The `conftest.py` in Pytest contains shared fixtures for all the tests. 

`LLMMocker` is used to mock the responses from LLM. `llm_mocker` pytest fixture that uses `LLMMocker` to patch and return the mocked responses for a given prompt. This way other operations can use this fixture as following 

```python
@pytest.mark.asyncio
async def test_map_async(llm_mocker: Callable[[dict[str, str]], LLMMocker]) -> None:
    mocker = llm_mocker(
        {
            "What is 1 + 1?": "2",
            "What is 1 + 3?": "4",
            "What is 1 + 5?": "6",
        }
    )

    with mocker.patch_llm():
        result: list[str] = await map([1, 3, 5], "What is 1 + {:d}?")

    assert result == ["2", "4", "6"]
```

Overall, the code in this library was beautifully thought out. Some lessons I learned:
* Use Pydantic for validation and structured data.
* Manage concurrency with async `Condition` and `Semaphore`.
* Use overloading and generics to make type hints meaningful.
* Offer both sync and async interfaces for flexibility.
* Write unit tests that isolate LLM calls through fixtures and mocks.
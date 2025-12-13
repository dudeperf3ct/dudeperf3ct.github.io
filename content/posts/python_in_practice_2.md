---
author: [""]
title: "Python in practice"
date: 2025-12-25
summary: "Patterns in python code design"
description: ""
tags: ["python", "best-practices"]
ShowToc: false
ShowBreadCrumbs: true
---

I was reading the source code for [ArcticTraining](github.com/snowflakedb/ArcticTraining) library designed for post-training LLMs

## Registry

There is a `RegistryMeta` class that follows the [registry](https://github.com/faif/python-patterns/blob/master/patterns/behavioral/registry.py) pattern to keep track of all subclasses of a base class.

```python
class RegistryMeta(ABCMeta):
    _registry: dict[str, type] = {}
    
    def __new__(mcs: Type["RegistryMeta"], name: str, bases: Tuple, class_dict: Dict) -> Type:
        """Creates a new class, validates it, and registers it."""
        cls: Type = super().__new__(mcs, name, bases, class_dict)
        ...
        # Register subclass if not in the registry already
        mcs._registry[base_type][registry_name] = cls
```

How this is used in downstream tasks such as data sources, models?

**Data**



**Model**

Similar to data, models also have a base factory class.

## Callbacks

## Config driven design


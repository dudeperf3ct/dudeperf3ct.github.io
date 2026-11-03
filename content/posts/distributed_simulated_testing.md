---
author: [""]
title: "What is Deterministic Simulation Testing?"
date: 2026-11-03
summary: ""
description: "A dummy's guide to distributed simulation testing"
tags: ["distributed systems", "dst", "testing"]
ShowToc: false
ShowBreadCrumbs: true
---

Most applications start with a fairly simple architecture: an API service talking to a database. As the application grows, we start adding components such as a cache to speed things up, read replicas to distribute the load, perhaps a message queue for background tasks. Before we know it, we're staring at a monstrous architecture that's increasingly difficult to reason about. The final boss might involve a database sharded across regions, with replicas coordinating through a consensus algorithm to agree on the state of the system.

Building such a system is one challenge. Testing it is another. Each component might work perfectly in isolation, but what happens when a network request is delayed, a machine crashes halfway through an operation, or two nodes attempt to update the same state concurrently? The order in which these events occur can change the outcome, and reproducing the exact sequence that caused a bug can be incredibly difficult.

How does one think about testing any software application? There's no right answer. The software industry has evolved various testing approaches and these take some form of the following

- **Unit testing:** Testing individual functions or components in isolation.
- **Functional testing:** Verifying that software behaves according to its requirements.
- **Integration testing:** Testing how different components interact.
- **Smoke testing:** Quick checks to verify that critical functionality works.
- **Fuzz or property-based testing:** Generating varied inputs to discover failures or violations of expected properties.
- **End-to-end (E2E) testing:** Testing complete application workflows from start to finish.

The testing approaches above help cover both happy and unhappy paths that an application might encounter in production. However, as applications grow, the number of possible states and execution orderings can increase rapidly. A particular combination of network delays, failures, and concurrent operations might be difficult to trigger, let alone reproduce, even with comprehensive test coverage.

There is another testing approach, determinisitic simulation testing (DST) that I came across reading about TigerBeetle. 

> [!info]
> TigerBeetle is a financial database. The engineering behind such a system is mind blowing. A shout out to Joran Greef for arguably the best [techxplainer video](https://www.youtube.com/watch?v=y2_BqkKTbD8) covering the concepts powering TigerBeetle.

As the name implies, deterministic simulation testing (DST) aims to make the execution of a simulated application deterministic and reproducible. In a real system, even with identical inputs, the execution can differ between runs. A network request might be delayed, a disk operation might complete earlier, or the operating system might schedule threads in a different order.

DST addresses this by bringing these sources of nondeterminism under the simulator's control. The real-world clock is replaced with a simulated (virtual) clock, while network I/O, disk I/O, and task scheduling are simulated. Unlike a real clock, the virtual clock can advance directly to the next scheduled event. For example, testing a 30-second timeout doesn't necessarily require waiting 30 seconds in real time. The simulator can inject failures, delays, timeouts, and other conditions, using randomness to explore different execution scenarios. These random decisions are derived from a fixed seed, allowing the same sequence to be reproduced.

If a particular seed triggers a bug, the same execution can be replayed to understand what went wrong, provided all relevant sources of nondeterminism are controlled. The simulator also checks certain properties or invariants to identify when the application behaves incorrectly. The simulated test environment runs checks or assertions against the application's behaviour.

Does this mean my URL shortener system requires DST? Probably not. Not every application needs this level of testing. DST becomes particularly useful when failures involving concurrency, shared state, and coordination are difficult to reproduce, and the cost of those failures justifies the additional engineering effort.

So, how does one implement DST? Imagine an application running across multiple machines, making network calls, reading and writing to disk, and executing concurrent operations. Sources of nondeterminism are sprinkled everywhere. Broadly, there are two approaches:

1. Application-aware simulation: Build the application so that dependencies such as clocks, network I/O, disk I/O, and scheduling can be replaced with simulated implementations during testing. FoundationDB and TigerBeetle follow this approach.
2. Deterministic virtualisation: Use the non-deterministic code in a deterministic hypervisor. [Antithesis](https://antithesis.com/) platform takes this approach.

The first approach is easier when simulation is considered from the beginning of application development. One way to simplify things is to use a single-threaded event loop that simulates multiple concurrent operations while controlling the order in which they execute.  With multiple threads or CPU cores, the operating system and hardware can introduce different execution orderings between runs, making deterministic replay harder. A single-threaded simulator avoids much of this complexity by controlling the scheduling itself.

For the second approaach, Antithesis team has a [great engineering blog](https://antithesis.com/blog/deterministic_hypervisor) detailing how they approached building a deterministic hypervisor. As the blog notes, the modern CPU is complex and creating a deterministic hypervisor is hard. The hypervisor controls sources of nondeterminism, including virtual time, CPU execution, and I/O.

Pierre Zemb has one of the [best resources](https://pierrezemb.fr/posts/learn-about-dst/) to get started with DST. There are many real-world scenarios mentioned in the blog. My favourites were the [Amp](https://wickstrom.tech/2025-08-28-findings-bugs-coding-agent-lightweight-dst.html) coding agent using lightweight DST in TypeScript and the blog on [finding bugs in Raft implementation](https://antithesis.com/blog/2026/finding-bugs-in-raft-implementations) by Antithesis. `etcd` uses Raft to coordinate its replicated state across etcd members, and Kubernetes relies on `etcd` for its cluster state.

So, what does this look like in practice? Not sure yet, I want to explore how DST might fit into ordinary application development.
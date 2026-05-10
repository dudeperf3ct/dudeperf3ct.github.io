---
author: [""]
title: "Peeling Back the Stack: A Toy Model of Stack Frames"
date: 2026-05-10
summary: ""
description: "Building a toy stack model to understand what goes on under hood."
tags: ["programming-language", "TIL"]
ShowToc: true
ShowBreadCrumbs: true
---

Learning programming languages is fun. My usual path is a couple of years writing the code and building projects in the particular language to get comfortable. It includes learning the best practices and understanding the different ways something could have been implemented. Then, occasionally, I get curious about what is happening under the hood sort of like peeling a layer of onion.

Eventually, peeling back enough layers, it reaches the compiler and finally the assembly or machine code. This code is then executed by the CPU.

Programs live in memory, and two important areas of memory are the stack and the heap. In this post, I will focus on the stack and build a toy model for understanding stack frames.

This post is inspired by Phrack volume 7 issue 49 on [Smashing the Stack For Fun and Profit](https://phrack.org/issues/49/14) and Signals and Thread podcast by Jane Street on [Memory Management](https://signalsandthreads.com/memory-management/). The article describes the working of the stack and how buffer overflowing can be used to exploit the system. The podcast provides comprehensive discussion around memory management particularly focusing on compilers, type system and garbage collection.

## Stack Toy Model

The basic programming exercise taught as part of every language is to implement a stack data structure. There are two important operations: `PUSH` and `POP`. `PUSH` operation adds an element to the top of the stack and `POP` operation removes the top element from the stack. The stack is a Last In First Out (LIFO) data structure, meaning that the last element added to the stack will be the first one to be removed. The CPU implements instructions to `PUSH` and `POP`. The stack size can also be dynamically adjusted by the kernel.

The Phrack article assumes an Intel x86 32-bit CPU running Linux and I will use the same for explaining the concepts and examples below.

- the stack grows downward, toward lower memory addresses
- one machine word is 4 bytes (32-bit)
- `ESP` is the Extended Stack Pointer: it points at the current top of the stack
- `EBP` is the Extended Base Pointer: it acts as the stable frame pointer for the current function
- `EIP` is the Extended Instruction Pointer: it points at the instruction currently being executed or the next instruction to execute
- function arguments are pushed onto the stack
- local variables live inside the called function's stack frame

**Stack Frame**

The stack frame contains local variables, return address, and other information needed for the function execution. A new stack frame is created for each function call, and it is destroyed when the function returns.

For example, consider the following slightly modified C code from the article

```c
void function(int a, int b, int c) {
   char buffer1[5];
   char buffer2[10];
}

int main() {
  function(1,2,3);
  return 0;
}
```

Here, the stack frame for the `function` contains

* Function arguments: `a`, `b`, `c` 
* Return address: the instruction in `main` immediately after the call to `function`
* the saved old frame pointer (more on this shortly)
* Local variables: `buffer1` and `buffer2`

It looks something like this

```
higher addresses

[ c              ]  EBP + 16
[ b              ]  EBP + 12
[ a              ]  EBP + 8
[ return address ]  EBP + 4
[ saved EBP      ]  EBP + 0   <-- EBP / frame pointer
[ buffer1        ]  EBP - 8
[ buffer2        ]  EBP - 20

lower addresses
```

When `main` calls `function`, the program needs to remember where to continue after `function` finishes. That location is stored as the return address.

The stack frame for `main` contains:

* Return address: the address to return to after `main` finishes. 

**Stack pointer (SP)**

Stack pointer points either to top of the stack or next available slot in the stack depending on the architecture. On 32-bit x86, this register is called `ESP`.

When a function is called, a new stack frame is created for that function, and the stack pointer is adjusted accordingly to point to the new top of the stack. 

Using the same example when `function` is called within `main` if we look at the assembly code,

```asm
main:
     ...

    pushl $3        ; push c
    pushl $2        ; push b
    pushl $1        ; push a
    call  function  ; push return address, then jump to function

    ...
```

Here, the arguments to the function are pushed onto the stack in reverse order and then return address is pushed on the stack when `call` instruction is executed.

**Frame pointer (FP)**

The frame pointer is a stable reference point inside the current function's stack frame. On 32-bit x86, this register is called `EBP`.

It is typically used to access local variables and parameters within the stack frame. The FP is set to the value of the stack pointer at the beginning of a function call (function prologue), and it remains unchanged throughout the execution of the function. This allows for easy access to local variables and parameters using fixed offsets from the FP.

The stack pointer could be used to access local variables but since the size of the stack frame can change during the execution of the function (e.g. due to variable length arrays or dynamic memory allocation), it can be more difficult to access local variables using the stack pointer. The frame pointer provides a stable reference point for accessing local variables and parameters, regardless of changes to the stack pointer.

For the above `function` in C code,

In the Phrack article's 32-bit assembly output, the start of `function` uses a function prologue like,

```asm
function:
    pushl %ebp          ; save caller's frame pointer
    movl  %esp, %ebp    ; create this function's frame pointer
    subl  $20, %esp     ; reserve space for local variables
```

The `20` bytes here is the article's rounded local storage size: 8 bytes for `buffer1` and 12 bytes for `buffer2`.

The first instruction saves the caller's frame pointer before `EBP` is repointed at the new function frame. This lets the epilogue restore the caller's frame pointer before returning.

The end of the function may use an epilogue (a sequence of instructions at the end of the function) like

```asm
    movl %ebp, %esp     ; remove local variables
    popl %ebp           ; restore caller's frame pointer
    ret                 ; return to saved return address
```

After the prologue, the stack frame looks approximately like the one shown earlier in the stack frame section.

```
higher addresses

[ c              ]  EBP + 16
[ b              ]  EBP + 12
[ a              ]  EBP + 8
[ return address ]  EBP + 4
[ saved EBP      ]  EBP + 0   <-- EBP / frame pointer
[ buffer1        ]  EBP - 8
[ buffer2        ]  EBP - 20

lower addresses
```

**Instruction Pointer (IP)**

Instruction pointer (also known as program counter) is a register that holds the address of the next instruction to execute. When a function is called, the instruction pointer is updated to point to the first instruction of the called function. When the function returns, the instruction pointer is updated to point to the return address stored in the stack frame.

**Toy Model**

All the components discussed above work together 

```c
void function(int a, int b, int c) {
   char buffer1[5];
   char buffer2[10];
}

int main() {
  function(1,2,3);
  return 0;
}
```

The assembly code can be generated with `gcc -m32 -O0 -fno-stack-protector -fno-pic -S sample.c -o sample.s`.


{{< collapse summary="**Assembly code**" >}}

```asm
	.file	"sample.c"
	.text
	.globl	function
	.type	function, @function
function:
.LFB0:
	.cfi_startproc
	pushl	%ebp
	.cfi_def_cfa_offset 8
	.cfi_offset 5, -8
	movl	%esp, %ebp
	.cfi_def_cfa_register 5
	subl	$16, %esp
	nop
	leave
	.cfi_restore 5
	.cfi_def_cfa 4, 4
	ret
	.cfi_endproc
.LFE0:
	.size	function, .-function
	.globl	main
	.type	main, @function
main:
.LFB1:
	.cfi_startproc
	pushl	%ebp
	.cfi_def_cfa_offset 8
	.cfi_offset 5, -8
	movl	%esp, %ebp
	.cfi_def_cfa_register 5
	pushl	$3
	pushl	$2
	pushl	$1
	call	function
	addl	$12, %esp
	movl	$0, %eax
	leave
	.cfi_restore 5
	.cfi_def_cfa 4, 4
	ret
	.cfi_endproc
.LFE1:
	.size	main, .-main
	.ident	"GCC: (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
	.section	.note.GNU-stack,"",@progbits
```

The Phrack article's old compiler output rounded the local storage to 20 bytes: 8 bytes for `buffer1` and 12 bytes for `buffer2`.

When I compile with GCC 13 using `-m32 -O0`, GCC emits `subl $16, %esp` instead. That is because the exact amount of stack space depends on compiler version, ABI rules, alignment, and optimization settings.

{{< /collapse >}}


The interactive diagram below walks through the instructions and shows how each stack slot appears.

{{< stack-toy-model >}}

The Phrack article uses this stack-frame model to explain why classic stack buffer overflows were dangerous. If a program writes past the end of a local stack buffer, nearby stack-frame data may be overwritten, including the saved frame pointer or return address. Historically, attackers used this class of bug to redirect control flow. Modern systems add many protections against this. The toy model is just useful for understanding the underlying idea.

## Garbage Collectors

The stack is only one part of the memory story. Garbage collectors (GC) are another fascinating component present in most of the programming languages. The basic premise of GC is to look for objects that are no longer referenced and freeing the memory. 

Different programming languages implement various types of single or multiple GC approaches such as reference counting, generational GC, mark and sweep GC or tracing GC. Python, for example, implements both reference counting and generational GC. Rust language on the other hand does not have a GC and instead relies on ownership, borrowing and lifetime rules to manage memory safely without the need for a GC.

The GC pauses are why some Python applications see p99 latency spikes as mentioned in the articles by [Delivery Hero](https://deliveryhero.jobs/blog/when-fast-is-too-slow-why-we-rewrote-our-java-ranking-api-in-python/) and [OLX](https://tech.olx.com/scaling-recommendations-service-at-olx-db4548813e3a?gi=104ecffc91dc) engineering teams.

## Wrap Up

I recommend learning one language and using it sufficiently in practice to know it with good depth other than just writing syntactically correct code. Once you have done that, pick a different language. This provides a comparison framework and appreciation on various topics that I hadn't thought of one language present in other. For me, it was Python and now I am having a blast learning Rust.

---
author: [""]
title: "Where vLLM Cold-Start Time Goes on GKE?"
date: 2026-04-11
summary: "Measuring vLLM cold-start bottlenecks on GKE and evaluating ways to reduce time to first request."
description: ""
tags: ["vllm", "k8s", "llm", "gke"]
ShowToc: true
ShowBreadCrumbs: true
---

Have you ever wondered what takes so long before the first request is served when deploying an LLM application using vLLM on Kubernetes? In this post, I will look at different components involved until the first request, measure them and optimize the bottlenecks.

> [!NOTE]
> If you want a deeper look inside vLLM itself, I recommend [Aleksa Gordic's](https://www.aleksagordic.com/) post on [Inside vLLM: Anatomy of a High-Throughput LLM Inference System](https://www.aleksagordic.com/blog/vllm).

## Methodology

I define startup time as the duration from deployment creation to the first successful response from `/v1/chat/completions`. Unless stated otherwise, all measurements are taken on a GKE cluster in `europe-west1` with a single L4 GPU node. I report end-to-end time along with a breakdown of image pull, model acquisition, model loading, engine initialization, and first-request latency.

## Infrastructure

For this setup, I use [OpenTofu](https://opentofu.org/) IaC to provision the infrastructure on GCP. I choose GCP because there are relatively fewer shenangians involved in getting a GPU instance. The setup has two main components: a GKE cluster and a GCS bucket.

Google Kubernetes Engine (GKE) is our environment for deploying LLM-backed applications. I wanted to use a setup closer to how LLMs are deployed in production, hence a GKE-based approach. As part of the cluster, a 1 x L4-GPU backed GPU instance node pool is created. This is where the LLM is hosted and ready to serve traffic.

[Mistral 7B Instruct v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) LLM is used to serve requests using the [vLLM](https://vllm.ai/) engine. For all experiments, I pin vLLM to version `0.18` using the [vllm/vllm-openai:v0.18.0](https://hub.docker.com/layers/vllm/vllm-openai/v0.18.0/images/sha256-96c7e88811a07030f27bc44cd71b9007258a15f130cfec2bb4ab057512238b05) image.

> [!CODE]
> The code repository walks you through all the steps required to get started, provisioning the infrastructure and reproducing the experiments.<br>
> Repository: https://github.com/dudeperf3ct/benchmark-vllm-startup

## Baseline

Once all the infrastructure is provisioned, we are all set to run our experiments. The first step is to establish a strong baseline. For baseline, I measure the total time required until the first request is served. We will also look into what different components are involved and the time taken by each of the components. 

Before starting the baseline benchmarking, I also collect network-related information. The information includes network bandwidth captured by pulling the weights for the Mistral model from Hugging Face Hub. It shows an effective rate of ~397 MB/s for a cluster in the `europe-west1` region. That number matters because remote artifact download is a major part of cold start, and it varies significantly by environment.

For baseline deployment, I use the `vllm serve` command to host and serve Mistral model. Once image is downloaded, `vllm` performs a few operations before starting the server to serve the requests. These operations include,

* **Initializing**: Initializes vLLM engine for the selected LLM. There are [lots of parameters](https://docs.vllm.ai/en/stable/configuration/engine_args/) that can be used to configure the asynchronous vLLM engine such as input dtype, maximum sequence length, configuring tensor or pipeline parallelism sizes, enabling cuda graphs and torch compile, and logging related flags.
* **Model downloading and loading**: It uses the configured model path to load the model and tokenizer from the storage. Or if the model weights are not present it downloads them from Hugging Face Hub.
* **Torch compile**: Using `torch.compile` is recommended. It provides an inference speedup out of the box without any code changes. The PyTorch compiler compiles the model into optimized kernels tailored to the hardware. For example, if we apply a matrix multiplication operation to input followed by activation, in normal or eager mode, the input would be read from memory, the kernels would be launched to apply these operations in the specified order of execution. For each operation, the output is stored and read from memory. This creates a memory bottleneck and adds kernel launch overheads. The PyTorch compiler uses kernel fusion and other optimizations to combine these operations into a single kernel execution thus reducing the bottleneck and overheads improving the hardware utilization. 
* **Cuda graphs**: [CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/) is a feature that reduces the kernel launch overheads. It does this by recording GPU operations as a graph and replaying the graph that enables multiple CUDA kernels to be executed through a single CPU launch. A kernel launch is the act of the CPU submitting a unit of work to the GPU. When many small kernels are launched independently, the submission cost and synchronization gaps add up and reduce effective utilization. 
* **Server startup**: Starting the vLLM server on port 8080 exposes endpoints such as `/health`, `ping`, `/metrics` and `v1/chat/completions`.

As part of this benchmarking process, we send an example payload to `v1/chat/completions` endpoint. The script records the time for all the different components and operations involved until the first request is served.

There are two baseline variants: cold and warm.

### Cold

A cold start is the very first deployment on a newly provisioned node. It pulls the [vllm/vllm-openai:v0.18.0](https://hub.docker.com/layers/vllm/vllm-openai/v0.18.0/images/sha256-96c7e88811a07030f27bc44cd71b9007258a15f130cfec2bb4ab057512238b05) runtime image first. The image takes up 28 GB of space and in this setup it takes roughly `236 seconds` to download the image onto the node.

### Warm

The GKE node caches the vLLM image pulled so for the subsequent restart of the deployment, the time to pull vLLM image is practically zero. In the warm run, we re-run the benchmarking script again.

A second easy win is avoiding repeated downloads of the LLM weights from Hugging Face Hub for each restart of the deployment. We pull the image once from the hub and store them in persistent storage. Other deployment can reuse the same weights for creating the server.

### Comparison

{{< plotly file="static/images/vllm_plots/baseline_cold_vs_warm_e2e.html" >}}

The end-to-end graph shows the total time taken until first request is served. It shows the time taken prepare the deployment, the vllm server and time for first request. The time for first request `0.059s` is negligible compared to the rest. We effectively get `2.5x` speedup for warm runs. This is because startup preparation becomes neligible as well as vllm image is already pulled on the node.

{{< plotly file="static/images/vllm_plots/baseline_cold_vs_warm_startup_breakdown.html" >}}

This graph shows the breakdown of the process involved in preparing and starting the deployment. The majority of time for cold runs is spent pulling the vllm image from Docker Hub registry. This is primarily bottlenecked by the network speed of the current environment and Docker Hub platform's network throughput and potentially how Docker works. There are optimized approaches that we will shortly visit that could potentially be used to speed up the container startup time.

{{< plotly file="static/images/vllm_plots/baseline_cold_vs_warm_vllm_breakdown.html" >}}

This graph shows further breakdown of each of the vllm components. The time taken to load model is less for warm runs as it uses already cached model. The cold run on the other hand downloads the model weights for the first run. The rest of the components take similar time across both the runs. The engine bootstrap covers all the processes required for vLLM engine initialization, argument parsing, config resolution, spawning processes and initializing distributed and NCCL setups.

One striking result is the weight-loading block: it was only `30.09s` in cold baseline but it's jumped to `66.1s` in the warm variant. The vLLM has to pull the model from PVC storage which for GKE is [`pd-balanced`](https://docs.cloud.google.com/compute/docs/disks/performance). This read IOPS per GiB for Balanced PD is 6 vs 30 for SSD PD. This process of loading weights could be sped up further by swapping `pd-balanced` with `pd-ssd`.

## Bucket synchornization

Bucket synchronization, we will compare the Hugging Face Hub's platform network throughput against cloud bucket storage (Google Cloud Storage bucket in our case). The idea here is that pulling model weights from object storage would be faster than whatever limits are set by Hugging Face Hub. A parallel download can be performed to fetch multiple model artifacts. 

The baseline for this comparison shows that downloading the model directly from Hugging Face Hub takes about `53 seconds`. We will test whether the bucket strategy can beat this number.

{{< plotly file="static/images/vllm_plots/bucket_sync_cold_e2e.html" >}}

The end-to-end result show the time required to serve first request in bucket synchronization case is larger than baseline cold run.

{{< plotly file="static/images/vllm_plots/bucket_sync_cold_model_acquisition.html" >}}

The comparison plot shows loading the weights from PVC takes longer than downloading them directly and moving those into memory for a cold baseline. Even though the time for pulling weights from HF Hub is dominated, the overhead of bucket synchronization is slightly larger.

{{< plotly file="static/images/vllm_plots/bucket_sync_cold_vllm_breakdown.html" >}}

vLLM breakdown shows the bucket synchronization approach is faster than cold run. But we have already paid the cost as startup overhead for downloading the weights in this approach.

> [!NOTE]
> Hugging Face Hub has rearchitected their platform with awesome engineering described in [blog post here](https://huggingface.co/blog/rearchitecting-uploads-and-downloads) to speed up the uploads and downloads transfers.

## Image streaming

The major cold-start bottleneck is the size of the runtime image itself. There are broadly 3 approaches to optimize the size and startup time for large OCI-compatible images. We want to reduce the size of large images so that they can be downloaded and extracted quickly.

### Multi-stage builds

This is a simple approach where we make the application footprint as small as possible. For compiled languages like Rust or Go, it's as easy as having a single binary with all the dependencies and a minimal base image. For interpreted languages like Python where all dependencies are downloaded in a virtual environment in the first stage. In the second stage, we copy only the environment and application code that are actually needed reducing the footprint.

```Dockerfile
FROM python:3.11
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "app.py"]
```

A multi-stage version of the same image looks like this:

```Dockerfile
# Build stage
FROM python:3.11 AS builder
WORKDIR /app
COPY . /app
RUN pip install --prefix=/install -r requirements.txt

# Final stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY --from=builder /app /app
CMD ["python", "app.py"]
```

### Trimming or slimming

The slim-based approaches take one step further in reducing the size of image by keeping only the necessary and essential files on the system. They typically use the `strace` system call to trace all the files required and accessed by the application and remove the rest thus reducing the footprint of the image. It can reduce the size of the original image by 30x. This approach requires thoroughly testing the application in the slimed version of the image to see everything is working as expected. 

Let's optimize the following `Dockerfile`, where `app.py` is `print("hello from python")`,

```Dockerfile
FROM python:3.11
WORKDIR /app
COPY app.py .
CMD ["python", "app.py"]
```

The [`slim`](https://github.com/slimtoolkit/slim) CLI provides commands to optimize the image along with other functionality such as inspection and debugging.

```bash
docker build -t py-example:latest .
# Create slim version of the docker image for the application
slim build --target py-example:latest --tag py-example:slim --http-probe=false --exec "python /app/app.py"
```

The original size of `1.1 GB` is cut down to `26.1 MB` in the slimmed version, reducing the size by 42x.

```bash
$ docker images

IMAGE              ID             DISK USAGE
py-example:latest  55c869965ecd        1.1GB
py-example:slim    e951a068da9d       26.1MB
```

### Lazy pull

[Stargz Snapshotter](https://github.com/containerd/stargz-snapshotter) is one such project that optimizes time to start the container. It uses a [Stargz](https://github.com/google/crfs) archive format which is a seekable tar.gz that can seek the archive and extract the file entries selectively. The way it works is instead of pulling and extracting every image layer before the container starts, the runtime fetches file data on demand as the process touches it. It can also profile file access and rearrange frequently accessed files in eStargz format so they are fetched earlier during startup. Here is a [blog](https://medium.com/nttlabs/startup-containers-in-lightning-speed-with-lazy-image-distribution-on-containerd-243d94522361) for further details on these formats.

This matters most for very large images where startup touches only a subset of files. In those cases, time to first request can improve even if the full image would still take a long time to download in the background.

Let's record the time taken to start the container for a `python:3.10` image

```bash
time nerdctl run --platform=amd64 ghcr.io/stargz-containers/python:3.10-org python3 -c 'print("hi")'

[...]
elapsed: 70.2s                                                                    total:  334.7  (4.8 MiB/s)
hi
4.19 user 
5.25 system 
1:10.69 elapsed
```

Using eStargz format for the same image,

```bash
time nerdctl --snapshotter=stargz run --platform=amd64 ghcr.io/stargz-containers/python:3.10-esgz python3 -c 'print("hi")'

[...]
elapsed: 8.8 s                                                                    total:  11.7 K (1.3 KiB/s)
hi
0.33 user 
0.11 system 
0:11.45 elapsed
```

That is an impressive speed up of 6.2x speed-up in end-to-end container startup time, reducing startup from `70.69s` to `11.45s`.

GKE provides [image streaming](docs.cloud.google.com/kubernetes-engine/docs/how-to/image-streaming) support for K8s workloads. There was no significant improvement in pulling the vLLM docker image. My current hypothesis is that image layout does not benefit from managed streaming as-is and that an eStargz-optimized image may be required. There are a couple of [open issues](https://github.com/orgs/vllm-project/projects/33) on vLLM project related to optimizing the Docker image, so this still looks like a promising approach.

## Run:ai model streamer

Run:ai uses a different approach for loading model weights. In the traditional flow, model weights are downloaded to storage, read into CPU memory, and then copied into GPU memory. That means the CPU is involved in multiple stages of the critical path.

Run:ai overlaps more of that work. It uses concurrent reads and dedicated CPU buffers so the application can continue loading tensors to GPU while other tensors are still being fetched from storage.

Quoting these from the [docs](https://github.com/run-ai/runai-model-streamer/tree/master) on how it works,

> The Streamer uses multiple threads to read tensors concurrently from a file in some file or object storage to a dedicated buffer in the CPU memory. Every tensor is given an identifier that subsequently is used by the application to load the tensor to the GPU memory. This way the application can load tensors from the CPU memory to the GPU memory while other tensors are being read from storage to the CPU memory.

> The model streaming utilizes OS-level concurrency to read data from local file systems, remote file systems, or object stores. The package employs a highly performant C++ layer to ensure maximum performance and minimum model load times, which is crucial for auto-scaling inference servers and keeping GPU idle times low.

{{< plotly file="static/images/vllm_plots/runai_warm_e2e.html" >}}

This comparison uses warm runs to isolate the model-loading path instead of runtime-image path. End to end, the warm baseline takes about `172s` while the Run:ai approach takes about `133s`, a reduction of roughly 23%.

{{< plotly file="static/images/vllm_plots/runai_warm_vllm_breakdown.html" >}}

The main gain comes from the model loading stage which drops from `68s` to `19s`.

## Wrap up

In this setup, the largest cold-start bottleneck was the container image pull, not the first request itself. Bucket synchronization did not improve end-to-end startup time, while Run:ai significantly reduced warm model-loading time. Image streaming still looks promising, but likely requires an image layout optimized for streaming.

# multi-thread-convolution# Convolution MPI Speedup

This project explores the acceleration of image convolution tasks using parallel computing. Three implementations were developed and evaluated: a sequential baseline (Naïve), a shared-memory model using OpenMP, and a distributed-memory model using MPI. The project benchmarks performance across various image sizes, demonstrating how workload size and hardware architecture influence parallelization effectiveness.

Developed as part of *COMS 4240: High-Performance Computing* at Iowa State University.

---

## Objectives

- Accelerate image convolution using edge detection filters (Sobel and Prewitt)
- Compare three implementations: Naïve, OpenMP, and MPI
- Evaluate execution time across small, medium, and large image sizes
- Analyze trade-offs between performance, scalability, and communication overhead
- Identify ideal use cases for shared vs. distributed memory models

---

## System Overview

| Module               | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| **Naïve Implementation** | Serial convolution using nested loops over image pixels                   |
| **OpenMP Implementation** | Multi-threaded shared-memory parallelization across image rows            |
| **MPI Implementation**    | Distributed convolution with inter-process communication via scatter/gather and row exchange |

---

## Tools & Libraries

- C/C++ with MPI and OpenMP
- Sobel and Prewitt edge detection filters
- `MPI_Scatterv`, `MPI_Sendrecv`, `MPI_Gatherv`
- Static scheduling for OpenMP parallelization

---

## Project Limitations

- Implementation assumes fixed 3×3 convolution kernels
- Edge pixels are not computed (due to padding limitations)
- MPI communication overhead increases on small images
- No hybrid (MPI + OpenMP) model implemented

---

## Execution Summary

| Image Size      | Best Performing Model | Observations                                     |
|------------------|------------------------|--------------------------------------------------|
| 900×550 / 1920×1080 | OpenMP                 | Low overhead, shared memory advantages           |
| 4800×6000 / 8000×8000 | MPI                    | Scalability and distributed memory reduce runtime |

> OpenMP scales well on moderate-size images and single machines.  
> MPI is ideal for large datasets that benefit from distributed memory.

---

## Trade-Offs Table

| Feature              | Naïve              | OpenMP                     | MPI                              |
|----------------------|--------------------|-----------------------------|----------------------------------|
| **Ease of Use**      | Simple             | Easy with minor refactoring | Complex setup with communication |
| **Performance**      | Poor (no scaling)  | Good on single machine       | Best on large datasets/multiple nodes |
| **Scalability**      | None               | Limited to machine cores     | Cross-machine scalability         |
| **Communication Overhead** | None        | None                        | High for small workloads         |

---

## License

MIT License — see `LICENSE` file.

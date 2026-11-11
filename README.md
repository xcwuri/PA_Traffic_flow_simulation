# Traffic Flow Simulation

A C++17 implementation of the Nagel–Schreckenberg traffic model on a one-lane ring road. It includes serial, OpenMP, MPI, and hybrid MPI+OpenMP modes, plus correctness verification and benchmark/plot tools.

## Build (Linux / WSL)

Install CMake, a C++ compiler, Open MPI, and Matplotlib for plotting:

```sh
sudo apt update
sudo apt install build-essential cmake libopenmpi-dev openmpi-bin python3-matplotlib
```

Configure and build all modes:

```sh
cmake -S . -B build-linux-mpi \
  -DCMAKE_CXX_COMPILER=mpicxx \
  -DTRAFFIC_ENABLE_MPI=ON \
  -DTRAFFIC_ENABLE_OPENMP=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-linux-mpi -j2
```

## Run and verify

Choose a mode with `--mode serial|openmp|mpi|hybrid`. `--verify` compares the final positions and speeds with the serial implementation using the same inputs.

```sh
# Serial reference
./build-linux-mpi/traffic_sim --mode serial --length 100000 --steps 1000 --density 0.2 --verify

# OpenMP, two threads
OMP_NUM_THREADS=2 ./build-linux-mpi/traffic_sim --mode openmp --length 100000 --steps 1000 --density 0.2 --verify

# MPI, four processes
mpirun --oversubscribe -np 4 ./build-linux-mpi/traffic_sim --mode mpi --length 100000 --steps 1000 --density 0.2 --verify

# Hybrid: four MPI processes, two OpenMP threads per process
OMP_NUM_THREADS=2 mpirun --oversubscribe -np 4 ./build-linux-mpi/traffic_sim --mode hybrid --length 100000 --steps 1000 --density 0.2 --verify
```

The model options include `--length`, `--steps`, `--density`, `--vmax`, `--slowdown`, `--seed`, and `--csv FILE`; run `--help` for details. `--csv` writes mean speed and flow per step. Omit it when benchmarking because it adds work and, in MPI modes, communication inside the timed loop.

Run the CMake correctness suite with:

```sh
ctest --test-dir build-linux-mpi --output-on-failure
```

The suite checks serial and OpenMP, MPI with 2 and 3 ranks, and hybrid with 2 ranks and 1 or 2 threads. Each test uses `--verify`.

## Benchmark and plots

`tools/benchmark.py` runs warmups and repeated measurements, saving individual timings to a raw CSV and medians/speedup/efficiency to a summary CSV. `tools/plot_results.py` creates graphs from the summary:

```sh
python3 tools/benchmark.py \
  --exe ./build-linux-mpi/traffic_sim \
  --output results/benchmark_500k_500.csv \
  --length 500000 --steps 500 --density 0.2 --seed 42 \
  --repeats 5 --warmups 1 --threads 1,2 --ranks 1,2 \
  --modes serial,openmp,mpi,hybrid --oversubscribe \
  --omp-wait-policy PASSIVE

python3 tools/plot_results.py results/benchmark_500k_500.csv \
  --output results/performance_500k_500.png
```

Saved raw samples, summaries, and graphs are in `results/`. The reported speedup is serial median time divided by parallel median time; efficiency is speedup divided by the requested worker count (`ranks × threads`). MPI timings synchronize ranks and use the slowest rank's elapsed time. Initialization, final gathering, and verification are excluded from timing.

## Results and limitations

The MPI-enabled Release build was verified on WSL2 with GCC 13.3.0, CMake 3.28.3, Open MPI 4.1.6, and OpenMP 4.5. CTest passed **7/7 checks**. An additional hybrid edge case with four ranks, two threads per rank, and only two cars also passed.

| Workload | Serial median | OpenMP (2 threads) | MPI (2 ranks) | Hybrid (2 ranks × 1 thread) | Hybrid (2 ranks × 2 threads) |
|---|---:|---:|---:|---:|---:|
| 100,000 cells × 300 steps | 0.035610 s | 1.079× | 2.183× | 2.158× | 2.052× |
| 500,000 cells × 500 steps | 0.306006 s | 1.637× | 2.257× | 2.099× | 2.211× |

These measurements were collected on a WSL2 allocation exposing two logical CPUs (one physical core with two hardware threads). The four-worker hybrid run oversubscribes that allocation; results describe this host and are not portable performance claims. The MPI implementation initializes the full state on rank 0 before scattering vehicle blocks.

## Project files

- `src/traffic_sim.cpp` — simulation, parallel modes, verification, metrics, and timing.
- `CMakeLists.txt` — build options and CTest correctness cases.
- `tools/benchmark.py`, `tools/plot_results.py` — measurement and graph generation.
- `results/` — saved benchmark samples, summaries, and plots.

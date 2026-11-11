// Main program: simulates traffic serially, with OpenMP, with MPI, or both.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef TRAFFIC_USE_MPI
#include <mpi.h>
#endif

using namespace std;

namespace {

struct Config {
    int length = 100000;
    int steps = 1000;
    int vmax = 5;
    double density = 0.20;
    double slowdown = 0.15;
    uint64_t seed = 42;
    string mode = "serial";
    string csv_path;
    bool verify = false;
};

struct State {
    vector<int> positions;
    vector<int> speeds;
};

struct Metrics {
    int step;
    double mean_speed;
    double flow;
};

uint64_t mix64(uint64_t x) {
    // Avalanche the input bits so nearby IDs/seeds produce unrelated-looking values.
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

double uniform01(uint64_t seed, uint64_t vehicle, uint64_t step) {
    // A stateless draw keyed by global vehicle ID and step makes the stochastic
    // slowdown identical in every execution mode and independent of scheduling.
    // Keep 53 high bits, which map exactly to the precision of a double fraction.
    const uint64_t bits = mix64(seed ^ mix64(vehicle + 0x632be59bd9b4e019ULL) ^ mix64(step));
    return static_cast<double>(bits >> 11) * (1.0 / 9007199254740992.0);
}

Config parse_args(int argc, char** argv) {
    // Read flag/value pairs, then reject invalid values before simulation begins.
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        const string arg = argv[i];
        auto value = [&](const char* flag) -> string {
            if (i + 1 >= argc) throw invalid_argument(string("Missing value after ") + flag);
            return argv[++i];
        };
        if (arg == "--length") cfg.length = stoi(value("--length"));
        else if (arg == "--steps") cfg.steps = stoi(value("--steps"));
        else if (arg == "--vmax") cfg.vmax = stoi(value("--vmax"));
        else if (arg == "--density") cfg.density = stod(value("--density"));
        else if (arg == "--slowdown") cfg.slowdown = stod(value("--slowdown"));
        else if (arg == "--seed") cfg.seed = stoull(value("--seed"));
        else if (arg == "--mode") cfg.mode = value("--mode");
        else if (arg == "--csv") cfg.csv_path = value("--csv");
        else if (arg == "--verify") cfg.verify = true;
        else if (arg == "--help" || arg == "-h") {
            cout << "Traffic Flow Simulation (Nagel-Schreckenberg cellular automaton)\n"
                      << "  --mode serial|openmp|mpi|hybrid  --length CELLS  --steps N\n"
                      << "  --density 0..1  --vmax N  --slowdown 0..1  --seed N\n"
                      << "  --verify  --csv FILE\n";
            exit(0);
        } else throw invalid_argument("Unknown argument: " + arg);
    }
    if (cfg.length < 2 || cfg.steps < 0 || cfg.vmax < 1)
        throw invalid_argument("length must be >= 2, steps >= 0, and vmax >= 1");
    if (!(cfg.density > 0.0 && cfg.density <= 1.0) || !isfinite(cfg.density))
        throw invalid_argument("density must be in (0, 1]");
    if (!(cfg.slowdown >= 0.0 && cfg.slowdown <= 1.0) || !isfinite(cfg.slowdown))
        throw invalid_argument("slowdown must be in [0, 1]");
    if (cfg.mode != "serial" && cfg.mode != "openmp" && cfg.mode != "mpi" && cfg.mode != "hybrid")
        throw invalid_argument("mode must be serial, openmp, mpi, or hybrid");
#ifndef _OPENMP
    if (cfg.mode == "openmp" || cfg.mode == "hybrid")
        throw invalid_argument("OpenMP mode requested, but this build has no OpenMP support");
#endif
#ifndef TRAFFIC_USE_MPI
    if (cfg.mode == "mpi" || cfg.mode == "hybrid")
        throw invalid_argument("MPI mode requested, but this build has no MPI support; configure with TRAFFIC_ENABLE_MPI=ON");
#endif
    return cfg;
}

State make_initial_state(const Config& cfg) {
    State s;
    // Cars are appended in cell order. The synchronous braking rule prevents
    // them from passing, so this cyclic vehicle order remains valid forever.
    // Scan cells in road order; the resulting array index is the stable vehicle ID.
    for (int cell = 0; cell < cfg.length; ++cell) {
        if (uniform01(cfg.seed, static_cast<uint64_t>(cell), 0) < cfg.density) {
            s.positions.push_back(cell);
            const int speed = static_cast<int>(mix64(cfg.seed + static_cast<uint64_t>(cell)) %
                                               static_cast<uint64_t>(cfg.vmax + 1));
            s.speeds.push_back(speed);
        }
    }
    if (s.positions.empty()) {
        s.positions.push_back(static_cast<int>(mix64(cfg.seed) % static_cast<uint64_t>(cfg.length)));
        s.speeds.push_back(0);
    }
    return s;
}

int gap_to_next(const vector<int>& positions, size_t i, int next_position, int length) {
    // Convert a wrapped position difference into a positive distance around the ring.
    int gap = next_position - positions[i];
    if (gap <= 0) gap += length;
    return gap;
}

int updated_speed(int old_speed, int gap, int vmax, double slowdown,
                  uint64_t seed, size_t vehicle, int step) {
    // Nagel–Schreckenberg update: accelerate, brake to the next-car gap, then
    // apply random slowdown. Movement happens only after all cars are updated.
    int v = min(old_speed + 1, vmax);
    v = min(v, gap - 1);
    if (v > 0 && uniform01(seed, vehicle, static_cast<uint64_t>(step + 1)) < slowdown) --v;
    return v;
}

void advance_serial(State& state, const Config& cfg, int step, bool parallel) {
    const size_t n = state.positions.size();
    // Separate output arrays preserve synchronous updates: every car reads only old state.
    vector<int> new_positions(n), new_speeds(n);
#ifdef _OPENMP
    if (parallel) {
        // Each iteration reads the immutable old state and owns exactly one
        // output slot. The implicit loop barrier completes the step before swap.
#pragma omp parallel for schedule(static)
        for (long long raw = 0; raw < static_cast<long long>(n); ++raw) {
            const auto i = static_cast<size_t>(raw);
            const size_t next = (i + 1) % n;
            const int gap = n == 1 ? cfg.length : gap_to_next(state.positions, i, state.positions[next], cfg.length);
            const int v = updated_speed(state.speeds[i], gap, cfg.vmax, cfg.slowdown, cfg.seed, i, step);
            new_speeds[i] = v;
            new_positions[i] = (state.positions[i] + v) % cfg.length;
        }
    } else
#endif
    {
        for (size_t i = 0; i < n; ++i) {
            const size_t next = (i + 1) % n;
            const int gap = n == 1 ? cfg.length : gap_to_next(state.positions, i, state.positions[next], cfg.length);
            const int v = updated_speed(state.speeds[i], gap, cfg.vmax, cfg.slowdown, cfg.seed, i, step);
            new_speeds[i] = v;
            new_positions[i] = (state.positions[i] + v) % cfg.length;
        }
    }
    state.positions.swap(new_positions);
    state.speeds.swap(new_speeds);
}

Metrics measure(const State& state, const Config& cfg, int step) {
    double total = 0.0;
    for (int v : state.speeds) total += v;
    const double mean = state.speeds.empty() ? 0.0 : total / static_cast<double>(state.speeds.size());
    return {step, mean, static_cast<double>(state.speeds.size()) * mean / cfg.length};
}

bool same_state(const State& a, const State& b) {
    return a.positions == b.positions && a.speeds == b.speeds;
}

#ifdef TRAFFIC_USE_MPI
void make_partition(int n, int size, vector<int>& counts, vector<int>& displs) {
    // Contiguous, nearly even vehicle-ID blocks are owned by ranks. Some
    // ranks may own zero cars when the process count exceeds the car count.
    counts.resize(size);
    displs.resize(size);
    // counts[r] is rank r's car count; displs[r] is its starting global vehicle ID.
    for (int r = 0; r < size; ++r) {
        const int begin = static_cast<int>((static_cast<long long>(n) * r) / size);
        const int end = static_cast<int>((static_cast<long long>(n) * (r + 1)) / size);
        displs[r] = begin;
        counts[r] = end - begin;
    }
}

State mpi_initialize(const Config& cfg, int rank, int size, vector<int>& counts,
                     vector<int>& displs) {
    // Rank 0 creates the ordered global state, then scatters each rank's
    // exclusive block. No rank stores another rank's vehicles during updates.
    State full;
    int n = 0;
    if (rank == 0) {
        full = make_initial_state(cfg);
        n = static_cast<int>(full.positions.size());
    }
    // All ranks need the global car count before they can compute matching partitions.
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    make_partition(n, size, counts, displs);
    const int local_n = counts[rank];
    State local{vector<int>(local_n), vector<int>(local_n)};
    // Scatter positions and speeds using identical ranges so IDs stay aligned with state.
    MPI_Scatterv(rank == 0 ? full.positions.data() : nullptr, counts.data(), displs.data(), MPI_INT,
                 local.positions.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? full.speeds.data() : nullptr, counts.data(), displs.data(), MPI_INT,
                 local.speeds.data(), local_n, MPI_INT, 0, MPI_COMM_WORLD);
    return local;
}

void advance_mpi(State& local, const Config& cfg, int step, int rank, int size,
                 const vector<int>& counts, const vector<int>& displs,
                 int total_cars, bool parallel) {
    const int local_n = static_cast<int>(local.positions.size());
    vector<int> new_positions(local_n), new_speeds(local_n);
    if (local_n > 0) {
        // Empty ranks do not join the neighbor exchange; active ranks find the
        // nearest owners with cars so the circular vehicle order has no gaps.
        int previous_active = rank;
        int next_active = rank;
        if (size > 1) {
            for (int offset = 1; offset < size; ++offset) {
                const int candidate = (rank - offset + size) % size;
                if (counts[candidate] > 0) { previous_active = candidate; break; }
            }
            for (int offset = 1; offset < size; ++offset) {
                const int candidate = (rank + offset) % size;
                if (counts[candidate] > 0) { next_active = candidate; break; }
            }
        }
        int next_first_position = local.positions.front();
        if (next_active != rank) {
            // Exchange one position with the next nonempty owner. Sending our
            // first position to the previous owner gives that rank its halo;
            // empty ranks are skipped so they cannot break the ring exchange.
            MPI_Sendrecv(&local.positions.front(), 1, MPI_INT, previous_active, 91,
                         &next_first_position, 1, MPI_INT, next_active, 91,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
#ifdef _OPENMP
        // MPI exchanges the halo on the main thread; OpenMP then assigns each
        // rank's disjoint output slots to threads. All threads read old state.
#pragma omp parallel for schedule(static) if(parallel)
#else
        (void)parallel;
#endif
        // Local neighbors come from this rank's block; only the last car uses
        // the exchanged halo position from the next nonempty rank.
        for (int j = 0; j < local_n; ++j) {
            const int next_position = (j + 1 < local_n) ? local.positions[j + 1] : next_first_position;
            const int gap = (total_cars == 1)
                                ? cfg.length
                                : gap_to_next(local.positions, static_cast<size_t>(j), next_position, cfg.length);
            const int v = updated_speed(local.speeds[j], gap, cfg.vmax, cfg.slowdown, cfg.seed,
                                        static_cast<size_t>(displs[rank] + j), step);
            new_speeds[j] = v;
            new_positions[j] = (local.positions[j] + v) % cfg.length;
        }
    }
    local.positions.swap(new_positions);
    local.speeds.swap(new_speeds);
}

State mpi_gather(const State& local, const vector<int>& counts,
                 const vector<int>& displs, int rank) {
    // Reassemble rank-owned slices in vehicle-ID order on rank 0.
    const int n = counts.empty() ? 0 : displs.back() + counts.back();
    State full;
    if (rank == 0) {
        full.positions.resize(n);
        full.speeds.resize(n);
    }
    MPI_Gatherv(local.positions.data(), static_cast<int>(local.positions.size()), MPI_INT,
                rank == 0 ? full.positions.data() : nullptr, counts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(local.speeds.data(), static_cast<int>(local.speeds.size()), MPI_INT,
                rank == 0 ? full.speeds.data() : nullptr, counts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);
    return full;
}
#endif

void write_csv(const string& path, const vector<Metrics>& rows) {
    // Save one row per completed step for optional visualization or analysis.
    ofstream out(path);
    if (!out) throw runtime_error("Could not open CSV output: " + path);
    out << "step,mean_speed,flow\n" << setprecision(10);
    for (const auto& row : rows) out << row.step << ',' << row.mean_speed << ',' << row.flow << '\n';
}

} // namespace

int main(int argc, char** argv) {
#ifdef TRAFFIC_USE_MPI
    int mpi_thread_level = MPI_THREAD_SINGLE;
    // MPI work stays on the main thread; OpenMP threads are used only for the
    // rank-local update loop after communication is complete.
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpi_thread_level);
    if (mpi_thread_level < MPI_THREAD_FUNNELED) {
        cerr << "MPI implementation does not provide MPI_THREAD_FUNNELED\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    int rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#else
    int rank = 0, world_size = 1;
#endif
    int result = 0;
    try {
        Config cfg = parse_args(argc, argv);
        const bool distributed = cfg.mode == "mpi" || cfg.mode == "hybrid";
        State state;
        vector<int> counts, displs;
        if (distributed) {
#ifdef TRAFFIC_USE_MPI
            state = mpi_initialize(cfg, rank, world_size, counts, displs);
#endif
        } else if (rank == 0) {
            state = make_initial_state(cfg);
        }
        int cars = static_cast<int>(state.positions.size());
#ifdef TRAFFIC_USE_MPI
        if (distributed && rank == 0) cars = displs.back() + counts.back();
#endif
        int total_cars = cars;
#ifdef TRAFFIC_USE_MPI
        if (distributed) MPI_Bcast(&total_cars, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
        vector<Metrics> trace;
#ifdef TRAFFIC_USE_MPI
        if (distributed) MPI_Barrier(MPI_COMM_WORLD);
#endif
        // Initialization is complete. MPI ranks start together; the ending
        // barrier includes the slowest rank's final work. The following max
        // reduction and final gather are outside the measured interval.
        // Only simulation steps are timed; initialization and final verification are excluded.
        const auto start = chrono::steady_clock::now();
        if (distributed) {
#ifdef TRAFFIC_USE_MPI
            for (int step = 0; step < cfg.steps; ++step) {
                advance_mpi(state, cfg, step, rank, world_size, counts, displs,
                            total_cars, cfg.mode == "hybrid");
                if (!cfg.csv_path.empty()) {
                    State full = mpi_gather(state, counts, displs, rank);
                    if (rank == 0) trace.push_back(measure(full, cfg, step + 1));
                }
            }
#endif
        } else if (rank == 0) {
            for (int step = 0; step < cfg.steps; ++step) {
                advance_serial(state, cfg, step, cfg.mode == "openmp");
                if (!cfg.csv_path.empty()) trace.push_back(measure(state, cfg, step + 1));
            }
        }
#ifdef TRAFFIC_USE_MPI
        if (distributed) MPI_Barrier(MPI_COMM_WORLD);
#endif
        const auto stop = chrono::steady_clock::now();
        double elapsed = chrono::duration<double>(stop - start).count();
#ifdef TRAFFIC_USE_MPI
        if (distributed) {
            double max_elapsed = 0.0;
            MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            elapsed = max_elapsed;
            state = mpi_gather(state, counts, displs, rank);
        }
#endif
        if (rank == 0) {
            if (!cfg.csv_path.empty()) write_csv(cfg.csv_path, trace);
            const Metrics final = measure(state, cfg, cfg.steps);
            cout << fixed << setprecision(6)
                      << "mode=" << cfg.mode << " length=" << cfg.length << " cars=" << total_cars
                      << " steps=" << cfg.steps << " vmax=" << cfg.vmax << " density=" << cfg.density
                      << " slowdown=" << cfg.slowdown << " seed=" << cfg.seed << '\n'
                      << "ranks=" << (distributed ? world_size : 1);
#ifdef _OPENMP
            cout << " threads=" << ((cfg.mode == "openmp" || cfg.mode == "hybrid") ? omp_get_max_threads() : 1);
#else
            cout << " threads=1";
#endif
            cout << " elapsed_seconds=" << setprecision(9) << elapsed
                      << setprecision(6)
                      << " updates_per_second=" << (elapsed > 0.0 ? static_cast<double>(total_cars) * cfg.steps / elapsed : 0.0)
                      << " mean_speed=" << final.mean_speed << " flow=" << final.flow << '\n';
            if (cfg.verify) {
                // Recompute with the serial reference, then compare exact ordered state.
                State reference = make_initial_state(cfg);
                for (int step = 0; step < cfg.steps; ++step) advance_serial(reference, cfg, step, false);
                const bool passed = same_state(state, reference);
                cout << "verification=" << (passed ? "PASS" : "FAIL") << '\n';
                if (!passed) result = 2;
            }
            if (!cfg.csv_path.empty()) cout << "csv=" << cfg.csv_path << '\n';
        }
#ifdef TRAFFIC_USE_MPI
        if (distributed) MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
#endif
    } catch (const exception& ex) {
        if (rank == 0) cerr << "error: " << ex.what() << '\n';
#ifdef TRAFFIC_USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }
    return result;
}

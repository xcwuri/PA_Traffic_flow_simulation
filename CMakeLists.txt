# Build configuration and correctness tests for the traffic simulator.
cmake_minimum_required(VERSION 3.16)
project(traffic_flow_simulation LANGUAGES CXX)

option(TRAFFIC_ENABLE_MPI "Build MPI and hybrid modes" OFF)
option(TRAFFIC_ENABLE_OPENMP "Build OpenMP mode when available" ON)

add_executable(traffic_sim src/traffic_sim.cpp)
target_compile_features(traffic_sim PRIVATE cxx_std_17)
if(MSVC)
  target_compile_options(traffic_sim PRIVATE /W4)
else()
  target_compile_options(traffic_sim PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()

if(TRAFFIC_ENABLE_OPENMP)
  # OpenMP is optional; without it, the executable still supports serial mode.
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(traffic_sim PRIVATE OpenMP::OpenMP_CXX)
  endif()
endif()

if(TRAFFIC_ENABLE_MPI)
  # MPI adds the distributed modes and is required when explicitly requested.
  find_package(MPI REQUIRED COMPONENTS CXX)
  target_link_libraries(traffic_sim PRIVATE MPI::MPI_CXX)
  # This project uses the C MPI API. Disable obsolete vendor C++ bindings
  # (which otherwise trigger warnings with current C++ compilers).
  target_compile_definitions(traffic_sim PRIVATE
    TRAFFIC_USE_MPI=1 OMPI_SKIP_MPICXX=1 MPICH_SKIP_MPICXX=1)
endif()

enable_testing()
# Tests invoke the program with --verify so parallel states must match serial.
add_test(NAME sequential_smoke
  COMMAND traffic_sim --mode serial --length 1000 --steps 20 --density 0.2 --verify)
if(OpenMP_CXX_FOUND)
  add_test(NAME openmp_1_thread_matches_sequential
    COMMAND traffic_sim --mode openmp --length 1000 --steps 20 --density 0.2 --verify)
  set_tests_properties(openmp_1_thread_matches_sequential PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=1")
  add_test(NAME openmp_2_threads_match_sequential
    COMMAND traffic_sim --mode openmp --length 1000 --steps 20 --density 0.2 --verify)
  set_tests_properties(openmp_2_threads_match_sequential PROPERTIES
    ENVIRONMENT "OMP_NUM_THREADS=2;OMP_DYNAMIC=FALSE")
endif()

if(TRAFFIC_ENABLE_MPI)
  foreach(ranks IN ITEMS 2 3)
    add_test(NAME mpi_${ranks}_ranks_match_sequential
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${ranks}
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:traffic_sim> ${MPIEXEC_POSTFLAGS}
        --mode mpi --length 2000 --steps 40 --density 0.22 --seed 17 --verify)
    set_tests_properties(mpi_${ranks}_ranks_match_sequential PROPERTIES
      PROCESSORS ${ranks}
      ENVIRONMENT "OMPI_MCA_rmaps_base_oversubscribe=1")
  endforeach()

  foreach(threads IN ITEMS 1 2)
    add_test(NAME hybrid_2_ranks_${threads}_threads_match_sequential
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2
        ${MPIEXEC_PREFLAGS} $<TARGET_FILE:traffic_sim> ${MPIEXEC_POSTFLAGS}
        --mode hybrid --length 2000 --steps 40 --density 0.22 --seed 17 --verify)
    set_tests_properties(hybrid_2_ranks_${threads}_threads_match_sequential PROPERTIES
      PROCESSORS 2
      ENVIRONMENT "OMP_NUM_THREADS=${threads};OMP_DYNAMIC=FALSE;OMPI_MCA_rmaps_base_oversubscribe=1")
  endforeach()
endif()

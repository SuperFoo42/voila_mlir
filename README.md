# VOILA

Out of tree MLIR dialect for adaptive reprogramming database queries.

## Dependencies

### Required
- cmake >= 3.11
- bison >= 3.2
- gcc >= 9 / clang >= 8
- python (llvm tests)
- make
- liblzma (for benchmarks)
- openmp (basically optional, but currently required)
- lld (can be configured, but link times are really bad)
### Optional
- ccache
- ninja
- Doxygen

## Requirements

In order to build the compiler, you have to at least fulfill the requirements to build llvm. Make sure to limit the number of parallel linker or compile jobs. 
The default for linking is two jobs, which corresponds to roughly 32GB of free memory.

## Setup

For setup run:

```
./configure
```

The configuration script invokes cmake, creates a subdirectory called `build` and configures the project as well as
downloading the dependencies, such as llvm and RE/flex. To build the project with optimizations, set `CMAKE_BUILD_TYPE` to `Release`, e.g.

```
./configure -DCMAKE_BUILD_TYPE=Release
```

To build the project, change into the build directory and run the build command:

```
cd build
cmake --build
```

This builds the compiler along with tests and benchmarks.

## Benchmarks
TBD

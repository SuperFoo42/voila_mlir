# VOILA

Out of tree MLIR dialect for adaptive reprogramming database queries.

## Dependencies
- cmake >= 3.11
- bison >= 3.2
- RE/flex
- gcc >= 9 / clang >= 8
- ninja
- Doxygen

## Requirements
In order to build the compiler, you have to at least fulfill the requirements to build llvm, which requires at least 10GB of free disk sapace and 4GB of free RAM.

## Setup
For setup run:

```
./configure
```

The configuration script invokes cmake, creates a subdirectory called `build` and configures the project. To build the project:
```
cd build
cmake --build
```

All further is TBD
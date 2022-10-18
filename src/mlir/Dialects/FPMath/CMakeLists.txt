add_mlir_dialect_library(MLIRFPMath
        IR/FPMathDialect.cpp
        IR/FPMathOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/mlir/Dialects/FPMath

        DEPENDS
        MLIRFPMathOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        range-v3
        fmt
        )
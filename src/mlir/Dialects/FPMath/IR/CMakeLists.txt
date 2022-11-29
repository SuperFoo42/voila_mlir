add_mlir_dialect_library(MLIRFPMath
        FPMathDialect.cpp
        FPMathOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/mlir/Dialects/FPMath

        DEPENDS
        MLIRFPMathOpsIncGen
        MLIRFPMathAttrIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        range-v3
        )
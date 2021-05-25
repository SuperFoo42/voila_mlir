add_mlir_dialect_library(MLIRVoila
        VoilaDialect.cpp
        VoilaOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/mlir

        DEPENDS
        MLIRVoilaOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )

add_subdirectory(lowering)
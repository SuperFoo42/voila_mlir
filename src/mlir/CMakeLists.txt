add_mlir_dialect_library(MLIRVoila
        VoilaDialect.cpp
        IR/VoilaOps.cpp
        Interfaces/VoilaInlinerInterface.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/mlir

        DEPENDS
        MLIRVoilaOpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )

add_library(voila_lowering)

target_link_libraries(voila_lowering fmt range-v3 MLIRVoila)

add_subdirectory(lowering)
add_subdirectory(Passes)
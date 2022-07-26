add_subdirectory(lowering)
add_subdirectory(Passes)
add_subdirectory(Interfaces)

add_mlir_dialect_library(MLIRVoila
        IR/VoilaDialect.cpp
        IR/VoilaOps.cpp
        ${VoilaInterfaces}


        ADDITIONAL_HEADER_DIRS
       ${PROJECT_SOURCE_DIR}/include/mlir/Dialects/Voila

        DEPENDS
        MLIRVoilaOpsIncGen
        MLIRPredicationOpInterfaceIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        range-v3
        fmt
        )


add_mlir_conversion_library(MLIRVoilaToInternal
        ${VoilaPasses}
        ${VoilaLowerings}

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/mlir/Dialects/Voila/Passes
        ${PROJECT_SOURCE_DIR}/include/mlir/Dialects/Voila/lowering

        LINK_LIBS PUBLIC
        MLIRVoila
        )

target_include_directories(MLIRVoilaToInternal PRIVATE ${PROJECT_SOURCE_DIR}/include/mlir/Dialects/Voila)
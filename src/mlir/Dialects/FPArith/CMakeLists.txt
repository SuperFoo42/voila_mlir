add_mlir_dialect_library(MLIRFPArith
        #VoilaDialect.cpp
        #IR/VoilaOps.cpp
        #Interfaces/VoilaInlinerInterface.cpp
        #Interfaces/PredicationOpInterface.cpp

        ADDITIONAL_HEADER_DIRS
        #../../../../include/mlir

        DEPENDS
        MLIRVoilaOpsIncGen
        MLIRPredicationOpInterfaceIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        )

#add_library(voila_lowering)

#target_link_libraries(voila_lowering range-v3 MLIRVoila fmt)

#add_subdirectory(lowering)
#add_subdirectory(Passes)
#add_subdirectory(Interfaces)
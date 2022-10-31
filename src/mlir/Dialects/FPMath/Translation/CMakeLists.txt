add_mlir_translation_library(MLIRFPMathToLLVMIRTranslation
        FPMathToLLVMIRTranslation.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/mlir/Dialects/FPMath

        DEPENDS
        MLIRFPMathOpsIncGen
        MLIRFPMathAttrIncGen

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRFPMath
        MLIRLLVMDialect
        MLIRSupport
        MLIRTargetLLVMIRExport
        )

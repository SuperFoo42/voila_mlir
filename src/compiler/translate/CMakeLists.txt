set(LLVM_LINK_COMPONENTS
        Support
        )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIROptLib
        MLIRCastInterfaces
        MLIRCallInterfaces
        MLIRSideEffectInterfaces
        MLIRAnalysis
        MLIRVoila
        MLIRVoilaToInternal
        MLIRPass
        MLIRSupport
        MLIRIR
        MLIRParser
        MLIRTransforms
        MLIRExecutionEngine
       # MLIRLLVMIR
        MLIRLLVMToLLVMIRTranslation
        MLIRTargetLLVMIRExport
        voila_compiler
        reflex
        spdlog
        magic_enum
        )

add_llvm_executable(voila-mlir-translate
        voila_translate.cpp
        )

target_compile_options(voila-mlir-translate PUBLIC
        -Wpedantic
        -Wall
        -Wextra
        -Wno-c++98-compat
        -Wno-c++98-c++11-c++14-compat
        -Wno-c++98-compat-pedantic
        -Wno-suggest-override)

target_compile_features(voila-mlir-translate PUBLIC cxx_std_20)
llvm_update_compile_flags(voila-mlir-translate)
target_link_libraries(voila-mlir-translate PRIVATE ${LIBS})

mlir_check_all_link_libraries(voila-mlir-translate)

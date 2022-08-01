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
        MLIRLLVMIR
        MLIRLLVMToLLVMIRTranslation
        MLIRTargetLLVMIRExport
        reflex
        spdlog
        magic_enum
        )

add_llvm_executable(voila-opt
        voila_opt.cpp
        )

target_compile_options(voila-opt PUBLIC
        -Wpedantic
        -Wall
        -Wextra
        -Wno-c++98-compat
        -Wno-c++98-c++11-c++14-compat
        -Wno-c++98-compat-pedantic
        -Wno-suggest-override)

target_compile_features(voila-opt PUBLIC cxx_std_20)
llvm_update_compile_flags(voila-opt)
target_link_libraries(voila-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(voila-opt)

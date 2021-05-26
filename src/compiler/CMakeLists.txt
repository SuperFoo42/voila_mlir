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
        MLIRPass
        MLIRSupport
        MLIRIR
        MLIRParser
        MLIRTransforms
        voila_compiler
        cxxopts
        fmt
        reflex
        )

add_llvm_executable(voila
        main.cpp
        )

target_compile_options(voila PUBLIC
        -Wpedantic
        -Wall
        -Wextra
        -Werror
        -Wno-c++98-compat
        -Wno-c++98-c++11-c++14-compat
        -Wno-c++98-compat-pedantic
        -Wno-suggest-override)

target_compile_features(voila PUBLIC cxx_std_20)
llvm_update_compile_flags(voila)

target_link_libraries(voila PRIVATE ${LIBS})

mlir_check_all_link_libraries(voila)

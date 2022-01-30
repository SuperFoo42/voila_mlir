target_sources(voila_lowering PRIVATE
        VoilaToLinalgLoweringPass.cpp
        VoilaToAffineLoweringPass.cpp
        VoilaToLLVMLoweringPass.cpp
        LinalgTiledLoopsToAffineForPass.cpp
        ParallelLoopToGpuMappingPass.cpp)
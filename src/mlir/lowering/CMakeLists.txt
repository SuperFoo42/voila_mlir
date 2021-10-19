add_library(voila_lowering
        VoilaToAffineLoweringPass.cpp
        VoilaToLLVMLoweringPass.cpp
        VoilaToLinalgLoweringPass.cpp
        VectorMultiReductionPass.cpp
        SelectOpLowering.cpp
        AvgOpLowering.cpp
        MaxOpLowering.cpp
        MinOpLowering.cpp
        SumOpLowering.cpp
        CountOpLowering.cpp
        EmitOpLowering.cpp
        NotOpLowering.cpp
        ReadOpLowering.cpp
        GatherOpLowering.cpp
        LoopOpLowering.cpp
        MoveOpLowering.cpp
        HashOpLowering.cpp
        InsertOpLowering.cpp
        LookupOpLowering.cpp
        )

target_link_libraries(voila_lowering fmt range-v3 MLIRVoila)
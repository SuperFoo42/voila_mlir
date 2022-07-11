target_sources(voila_lowering PRIVATE
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
        HashOpLowering.cpp
        InsertOpLowering.cpp
        LookupOpLowering.cpp
        ScatterOpLowering.cpp
)
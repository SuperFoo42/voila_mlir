#pragma once

namespace voila
{
    class Program;
    class TypeInferer;

    class TypeInferencePass
    {
        TypeInferer &inferer;

      public:
        explicit TypeInferencePass(TypeInferer &inferer) : inferer{inferer} {};

        TypeInferer &&inferTypes(Program &prog);
    };
} // namespace voila
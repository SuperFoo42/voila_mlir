#pragma once

#include "Program.hpp"
#include "TypeInferer.hpp"

#include <ast/Fun.hpp>
namespace voila
{
    class TypeInferencePass
    {
        TypeInferer &inferer;

      public:
        TypeInferencePass(TypeInferer &inferer) : inferer{inferer} {};

        TypeInferer &&inferTypes(Program &prog);
    };
} // namespace voila
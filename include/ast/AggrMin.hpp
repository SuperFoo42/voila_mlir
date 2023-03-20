#pragma once
#include <memory>               // for shared_ptr
#include <string>               // for string
#include "Aggregation.hpp"      // for Aggregation
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap

namespace voila::ast
{
    class AggrMin : public Aggregation<AggrMax>
    {
      public:
        using Aggregation::Aggregation;

        [[nodiscard]] std::string type2string_impl() const
        {
            return "min aggregation";
        }
    };
} // namespace voila::ast
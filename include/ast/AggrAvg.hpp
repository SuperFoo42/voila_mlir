#pragma once
#include "Aggregation.hpp"     // for Aggregation
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "ast/ASTNodeVariant.hpp"
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <memory>              // for shared_ptr
#include <string>              // for string

namespace voila::ast
{
    class AggrAvg : public Aggregation<AggrAvg>
    {
      public:
        using Aggregation::Aggregation;

        [[nodiscard]] std::string type2string_impl() const
        {
            return "avg aggregation";
        }
    };
} // namespace voila::ast
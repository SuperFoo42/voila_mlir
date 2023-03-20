#pragma once
#include "Aggregation.hpp"     // for Aggregation
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <memory>              // for shared_ptr
#include <string>              // for string

namespace voila::ast
{
    class AggrSum : public Aggregation<AggrSum>
    {
      public:
        using Aggregation::Aggregation;

        [[nodiscard]] std::string type2string_impl() const { return "sum aggregation"; }
    };
} // namespace voila::ast
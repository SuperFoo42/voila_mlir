#pragma once
#include "Aggregation.hpp"     // for Aggregation
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <memory>              // for shared_ptr
#include <string>              // for string

namespace voila::ast
{
    class AggrMax : public Aggregation<AggrMax>
    {
      public:
        using Aggregation::Aggregation;

        [[nodiscard]] std::string type2string_impl() const { return "max aggregation"; };
    };
} // namespace voila::ast
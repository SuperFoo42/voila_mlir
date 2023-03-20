#pragma once
#include "Aggregation.hpp"     // for Aggregation
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <memory>              // for shared_ptr
#include <string>              // for string

namespace voila::ast
{
    class AggrCnt : public Aggregation<AggrCnt>
    {
      public:
        using Aggregation::Aggregation;

        [[nodiscard]] std::string type2string_impl() const { return "count aggregation"; }
    };
} // namespace voila::ast
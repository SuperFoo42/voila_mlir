#pragma once
#include "BinaryOp.hpp"        // for Logical
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    class Or : public BinaryOp<Or>
    {
      public:
        using BinaryOp<Or>::BinaryOp;

        [[nodiscard]] std::string type2string_impl() const { return "or"; }
    };
} // namespace voila::ast
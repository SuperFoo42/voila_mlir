#pragma once

#include "BinaryOp.hpp"      // for Arithmetic
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    class Sub : public BinaryOp<Sub>
    {
      public:
        using BinaryOp<Sub>::BinaryOp;
        [[nodiscard]] std::string type2string_impl() const
        {
            return "sub";
        }
    };
} // namespace voila::ast

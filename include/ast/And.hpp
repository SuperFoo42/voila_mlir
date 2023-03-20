#pragma once
#include "BinaryOp.hpp"        // for Logical
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    class And : public BinaryOp<And>
    {
      public:
        using BinaryOp::BinaryOp;

        [[nodiscard]] std::string type2string_impl() const { return "and"; }
    };
} // namespace voila::ast
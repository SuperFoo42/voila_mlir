#pragma once
#include "BinaryOp.hpp"      // for Comparison
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    class Eq : public BinaryOp<Eq>
    {
      public:
        using BinaryOp<Eq>::BinaryOp;
        [[nodiscard]] std::string type2string_impl() const { return "eq"; }
    };
} // namespace voila::ast
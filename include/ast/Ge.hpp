#pragma once
#include <memory>               // for shared_ptr
#include <string>               // for string
#include <utility>              // for move
#include "BinaryOp.hpp"       // for Comparison
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap

namespace voila::ast
{
    class Ge : public BinaryOp<Ge>
    {
      public:
        using BinaryOp<Ge>::BinaryOp;
        [[nodiscard]] std::string type2string_impl() const { return "ge"; }
    };
} // namespace voila::ast
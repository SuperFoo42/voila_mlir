#pragma once
#include <memory>               // for shared_ptr
#include <string>               // for string
#include <utility>              // for move
#include "BinaryOp.hpp"       // for Comparison
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap

namespace voila::ast
{

    class Le : public BinaryOp<Le>
    {
      public:
        using BinaryOp<Le>::BinaryOp;
        [[nodiscard]] std::string type2string_impl() const { return "le"; }
    };
} // namespace voila::ast
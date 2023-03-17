#pragma once
#include <memory>               // for shared_ptr
#include <string>               // for string
#include <utility>              // for move
#include "Comparison.hpp"       // for Comparison
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap

namespace voila::ast
{
    class Ge : public Comparison
    {
      public:
        Ge(Location loc, ASTNodeVariant lhs, ASTNodeVariant rhs) : Comparison(loc, lhs, rhs)
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_ge() const final;

        Ge *as_ge() final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) final;
    };
} // namespace voila::ast
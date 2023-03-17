#pragma once
#include "Comparison.hpp"      // for Comparison
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    class Eq : public Comparison
    {
      public:
        Eq(Location loc, ASTNodeVariant lhs, ASTNodeVariant rhs) : Comparison(loc, lhs, rhs)
        {
            // TODO
        }
        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_eq() const final;

        Eq *as_eq() final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) final;
    };
} // namespace voila::ast
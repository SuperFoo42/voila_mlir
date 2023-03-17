#pragma once
#include "Arithmetic.hpp"      // for Arithmetic
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    // Expressions
    class Add : public Arithmetic
    {
      public:
        Add(const Location loc, ASTNodeVariant lhs, ASTNodeVariant rhs) : Arithmetic(loc, std::move(lhs), std::move(rhs))
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_add() const final;

        Add *as_add() final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) final;
    };
} // namespace voila::ast
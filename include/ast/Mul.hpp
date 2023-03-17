#pragma once
#include "Arithmetic.hpp"      // for Arithmetic
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{

    class Mul : public Arithmetic
    {
      public:
        Mul(const Location loc, ASTNodeVariant lhs, ASTNodeVariant rhs) : Arithmetic(loc, lhs, rhs)
        {
            // TODO
        }
        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_mul() const final;

        Mul *as_mul() final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) final;
    };
} // namespace voila::ast
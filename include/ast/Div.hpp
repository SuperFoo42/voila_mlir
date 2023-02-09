#pragma once
#include "Arithmetic.hpp"      // for Arithmetic
#include "Expression.hpp"      // for Expression
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    class ASTVisitor;

    class Div : public Arithmetic
    {
      public:
        Div(const Location loc, Expression lhs, Expression rhs) : Arithmetic(loc, std::move(lhs), std::move(rhs))
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_div() const final;

        Div *as_div() final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) final
        {
            return Arithmetic::clone<Div>(vmap);
        }
    };
} // namespace voila::ast
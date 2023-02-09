#pragma once
#include "Expression.hpp"      // for Expression
#include "Logical.hpp"         // for Logical
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    class ASTVisitor;

    class Not : public Logical
    {
        Expression mParam;

      public:
        explicit Not(const Location loc, Expression expr) : Logical(loc), mParam(std::move(expr))
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_not() const final;

        Not *as_not() final;

        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        [[nodiscard]] const Expression &param() const { return mParam; }
    };
} // namespace voila::ast
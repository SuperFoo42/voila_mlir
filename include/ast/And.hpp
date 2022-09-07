#pragma once
#include "ASTVisitor.hpp"
#include "Expression.hpp"
#include "Logical.hpp"

namespace voila::ast
{
    class And : public Logical
    {
        Expression mLhs, mRhs;

      public:
        And(const Location loc, Expression lhs, Expression rhs) :
            Logical(loc), mLhs{std::move(lhs)}, mRhs{std::move(rhs)}
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_and() const final;

        And *as_and() final;

        void visit(ASTVisitor &visitor) final;
        void visit(ASTVisitor &visitor) const final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        const Expression &lhs() const {
            return mLhs;
        }

        const Expression &rhs() const {
            return mRhs;
        }
    };
} // namespace voila::ast
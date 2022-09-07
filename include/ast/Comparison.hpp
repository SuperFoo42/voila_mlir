#pragma once
#include <utility>

#include "IExpression.hpp"
#include "Expression.hpp"
namespace voila::ast
{
    class Comparison : public IExpression
    {
        Expression mLhs;
        Expression mRhs;


      public:
        Comparison(const Location loc, Expression lhs, Expression rhs) : IExpression(loc), mLhs{std::move(lhs)}, mRhs{std::move(rhs)} {}
        [[nodiscard]] bool is_comparison() const final;

        Comparison *as_comparison() final;

        [[nodiscard]] std::string type2string() const override;

        void print(std::ostream &ostream) const final;

        [[nodiscard]] const Expression &lhs() const {
            return mLhs;
        }

        [[nodiscard]] const Expression &rhs() const {
            return mRhs;
        }

    protected:
        template <class T>
        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) requires std::is_base_of_v<Comparison, T> {
            return std::make_shared<T>(loc, mLhs.clone(vmap), mRhs.clone(vmap));
        }
    };
} // namespace voila::ast
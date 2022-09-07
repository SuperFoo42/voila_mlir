#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"

#include <utility>

namespace voila::ast
{
    class Aggregation : public IExpression
    {
        Expression mSrc;
        std::optional<Expression> mGroups;

      public:
        Aggregation(const Location loc, Expression col) : IExpression(loc), mSrc{std::move(col)}, mGroups(std::nullopt) {}
        Aggregation(const Location loc, Expression col, Expression groups) :
            IExpression(loc), mSrc{std::move(col)}, mGroups(std::move(groups))
        {
        }
        ~Aggregation() override = default;

        [[nodiscard]] bool is_aggr() const final;

        Aggregation *as_aggr() final;

        [[nodiscard]] std::string type2string() const override;

        void print(std::ostream &) const final {}

        [[nodiscard]] const Expression &src() const {
            return mSrc;
        }

        [[nodiscard]] const std::optional<Expression> &groups() const {
            return mGroups;
        }

    protected:
        template <class T>
        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) requires std::is_base_of_v<Aggregation, T>
        {
            if (mGroups.has_value())
                return std::make_shared<T>(loc, mSrc.clone(vmap), mGroups->clone(vmap));
            else
                return std::make_shared<T>(loc, mSrc.clone(vmap));
        }
    };
} // namespace voila::ast
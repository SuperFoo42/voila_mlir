#pragma once
#include "ASTVisitor.hpp"
#include "Expression.hpp"
#include "IExpression.hpp"

#include <string>
#include <vector>
#include <utility>

namespace voila::ast
{
    class Insert : public IExpression
    {
        Expression mKeys;
        std::vector<Expression> mValues;

      public:
        Insert(Location loc, Expression keys, std::vector<Expression> values) :
            IExpression(loc), mKeys{std::move(keys)}, mValues{std::move(values)}
        {
            // TODO
        }

        [[nodiscard]] bool is_insert() const final;

        Insert *as_insert() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        [[nodiscard]] const Expression &keys() const {
            return mKeys;
        }

        [[nodiscard]] const std::vector<Expression> &values() const {
            return mValues;
        }
    };
} // namespace voila::ast
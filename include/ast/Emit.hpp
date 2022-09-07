#pragma once
#include "ASTVisitor.hpp"
#include "Expression.hpp"
#include "IStatement.hpp"

#include <utility>
#include <vector>

namespace voila::ast
{
    class Emit : public IStatement
    {
        std::vector<Expression> mExprs;

      public:
        explicit Emit(Location loc, std::vector<Expression> expr) : IStatement(loc), mExprs{std::move(expr)} {}

        [[nodiscard]] bool is_emit() const final;

        Emit *as_emit() final;

        [[nodiscard]] std::string type2string() const final;

        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        [[nodiscard]] const std::vector<Expression> &exprs() const
        {
            return mExprs;
        }
    };

} // namespace voila::ast
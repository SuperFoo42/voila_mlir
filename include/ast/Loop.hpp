#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"
#include "Statement.hpp"
#include "ASTVisitor.hpp"
#include <utility>
#include <vector>

namespace voila::ast
{
    //TODO: fix this
    class Loop : public IStatement
    {
        Expression mPred;
        std::vector<Statement> mStms;

      public:
        Loop(const Location loc, Expression pred, std::vector<Statement> stms) : IStatement(loc), mPred{std::move(pred)}, mStms{std::move(stms)}
        {
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_loop() const final;

        Loop *as_loop() final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const override;
        void visit(ASTVisitor &visitor) override;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;
        // TODO
        // CrossingVariables crossing_variables;

        const Expression &pred() const {
            return mPred;
        }

        const std::vector<Statement> &stmts() const
        {
            return mStms;
        }

    };

} // namespace voila::ast
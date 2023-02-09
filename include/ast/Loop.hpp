#pragma once
#include "Expression.hpp"      // for Expression
#include "IStatement.hpp"      // for IStatement
#include "Statement.hpp"       // for Statement
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move
#include <vector>              // for vector

namespace voila::ast
{
    class ASTVisitor;

    // TODO: fix this
    class Loop : public IStatement
    {
        Expression mPred;
        std::vector<Statement> mStms;

      public:
        Loop(const Location loc, Expression pred, std::vector<Statement> stms)
            : IStatement(loc), mPred{std::move(pred)}, mStms{std::move(stms)}
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

        [[nodiscard]] const Expression &pred() const { return mPred; }

        [[nodiscard]] const std::vector<Statement> &stmts() const { return mStms; }
    };

} // namespace voila::ast
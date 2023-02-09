#pragma once
#include "Expression.hpp"      // for Expression
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "ast/IExpression.hpp" // for IExpression
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    class ASTVisitor;

    class Selection : public IExpression
    {
        Expression mParam;
        Expression mPred;

      public:
        explicit Selection(const Location loc, Expression expr, Expression pred)
            : IExpression(loc), mParam(std::move(expr)), mPred(std::move(pred))
        {
            // TODO
        }

        [[nodiscard]] bool is_select() const final;

        Selection *as_select() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        [[nodiscard]] const Expression &param() const { return mParam; }

        [[nodiscard]] const Expression &pred() const { return mPred; }
    };
} // namespace voila::ast
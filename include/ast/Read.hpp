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

    class Read : public IExpression
    {
        Expression mColumn, mIdx;

      public:
        Read(Location loc, Expression lhs, Expression rhs)
            : IExpression(loc), mColumn{std::move(lhs)}, mIdx{std::move(rhs)}
        {
            // TODO
        }

        [[nodiscard]] bool is_read() const final;

        Read *as_read() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        [[nodiscard]] const Expression &column() const { return mColumn; }

        [[nodiscard]] const Expression &idx() const { return mIdx; }
    };
} // namespace voila::ast
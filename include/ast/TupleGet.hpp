#pragma once
#include "Expression.hpp"
#include "IExpression.hpp"
#include "IntConst.hpp"

#include <cassert>
#include <utility>
#include "ASTVisitor.hpp"
namespace voila::ast
{
    class TupleGet : public IExpression
    {
      public:
        TupleGet(Location loc, Expression exp, intmax_t idx);

        [[nodiscard]] bool is_tuple_get() const final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        const Expression expr;
        const std::intmax_t idx;

        std::unique_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;
    };
} // namespace voila::ast
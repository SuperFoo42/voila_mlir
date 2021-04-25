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
        TupleGet(std::string exp, intmax_t idx);

        [[nodiscard]] bool is_tuple_get() const final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::string expr;
        intmax_t idx;
    };
} // namespace voila::ast
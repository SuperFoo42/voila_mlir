#pragma once
#include "Expression.hpp"
#include "IExpression.hpp"

#include <utility>
#include <vector>
#include "ASTVisitor.hpp"
namespace voila::ast
{
    class TupleCreate : public IExpression
    {
      public:
        explicit TupleCreate(std::vector<Expression> tupleElems);

        [[nodiscard]] bool is_tuple_create() const final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::vector<Expression> elems;
    };
} // namespace voila::ast
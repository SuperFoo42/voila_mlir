#pragma once
#include "Expression.hpp"
#include "IExpression.hpp"

#include <utility>
#include <vector>
namespace voila::ast
{
    class TupleCreate : public IExpression
    {
      public:
        explicit TupleCreate(std::vector<Expression> tupleElems);

        [[nodiscard]] bool is_tuple_create() const final;

        [[nodiscard]] std::string type2string() const override;
        void print(std::ostream &ostream) const override;

        std::vector<Expression> elems;
    };
} // namespace voila::ast
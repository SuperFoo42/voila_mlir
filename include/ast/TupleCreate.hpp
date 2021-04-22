#pragma once
#include "IExpression.hpp"
#include "Expression.hpp"
#include <vector>
namespace voila::ast
{
   class TupleCreate : IExpression
    {
      public:
        TupleCreate(std::vector<Expression> &tupleElems) : IExpression(), elems{tupleElems} {}

        bool is_tuple_create() const final
        {
            return true;
        }

        std::string type2string() const override
        {
            return "tuple create";
        }

        std::vector<Expression> elems;
    };
} // namespace voila::ast
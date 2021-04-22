#pragma once
#include "Expression.hpp"
#include "IExpression.hpp"
#include "IntConst.hpp"
#include <cassert>

namespace voila::ast
{
    class TupleGet : IExpression
    {
      public:
        TupleGet(const Expression &exp, const Expression &idx) : IExpression(), expr{exp}, idx{idx}
        {
            assert(idx.is_integer());
            this->idx = idx.as_integer();
            assert(this->idx->val >= 0);
            // TODO: check expr tuple and idx in range
        }

        bool is_tuple_get() const final
        {
            return true;
        }

        std::string type2string() const final
        {
            return "tuple get";
        }

        Expression expr;
        IntConst *idx;
    };
} // namespace voila::ast
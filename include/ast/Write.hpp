
#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"

#include <vector>

namespace voila::ast
{
    class Write : IStatement
    {
      public:
        Write(const Expression &dest_col, const Expression &wpos, const Expression &src_col) :
            IStatement(), dest{dest_col}, start{wpos}, src{src_col}
        {
        }

        bool is_write() const final
        {
            return true;
        }

        Write *as_write() final
        {
            return this;
        }

        std::string type2string() const final
        {
            return "write";
        }

        Expression dest;
        Expression start;
        Expression src;
    };

} // namespace voila::ast

#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"

#include <utility>
#include <vector>

namespace voila::ast
{
    class Write : public IStatement
    {
      public:
        Write(std::string dest_col, Expression wpos, std::string src_col);

        [[nodiscard]] bool is_write() const final;

        Write *as_write() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;

        std::string dest;
        Expression start;
        std::string src;
    };

} // namespace voila::ast
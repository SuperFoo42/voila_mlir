#pragma once
#include "Expression.hpp"
#include "IStatement.hpp"
#include "Statement.hpp"

#include <optional>
#include <utility>
#include <vector>

namespace voila::ast
{
    /**
     * @brief Meta node to wrap expressions into statements
     *
     */
    class StatementWrapper : public IStatement
    {
      public:
        explicit StatementWrapper(Expression expr);

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_statement_wrapper() const final;

        std::optional<Expression> as_expression() final;

      private:
        void print(std::ostream &ostream) const final;

      public:
        Expression expr;
    };

} // namespace voila::ast
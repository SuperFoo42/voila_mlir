#include "ast/IStatement.hpp"
#include <fmt/format.h>
namespace voila::ast
{
    void IStatement::set_predicate(Expression)
    {
        throw PredicationUnsupportedException(fmt::format("Predication not supported for: {}.", type2string()));
    }

    std::optional<Expression> IStatement::get_predicate()
    {
        throw PredicationUnsupportedException(fmt::format("Predication not supported for: {}.", type2string()));
    }

    bool IStatement::is_stmt() const
    {
        return true;
    }

    bool IStatement::is_loop() const
    {
        return false;
    }
    bool IStatement::is_assignment() const
    {
        return false;
    }
    bool IStatement::is_emit() const
    {
        return false;
    }
    bool IStatement::is_write() const
    {
        return false;
    }

    bool IStatement::is_function_call() const
    {
        return false;
    }

    bool IStatement::is_statement_wrapper() const
    {
        return false;
    }
    [[nodiscard]] StatementWrapper *IStatement::as_statement_wrapper()
    {
        return nullptr;
    }
    IStatement *IStatement::as_stmt()
    {
        return this;
    }

    Loop *IStatement::as_loop()
    {
        return nullptr;
    }
    Assign *IStatement::as_assignment()
    {
        return nullptr;
    }
    Emit *IStatement::as_emit()
    {
        return nullptr;
    }
    FunctionCall *IStatement::as_function_call()
    {
        return nullptr;
    }

    Write *IStatement::as_write()
    {
        return nullptr;
    }

    std::optional<Expression> IStatement::as_expression()
    {
        return std::nullopt;
    }
} // namespace voila::ast
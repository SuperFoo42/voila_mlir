#include "ast/StatementWrapper.hpp"

namespace voila::ast
{
    std::string StatementWrapper::type2string() const
    {
        return "statement wrapper";
    }
    bool StatementWrapper::is_statement_wrapper() const
    {
        return true;
    }
    std::optional<Expression> StatementWrapper::as_expression()
    {
        return std::make_optional(expr);
    }
    StatementWrapper::StatementWrapper(Expression expr) : IStatement(), expr{std::move(expr)} {}
    void StatementWrapper::print(std::ostream &) const
    {
    }

    void StatementWrapper::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void StatementWrapper::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast
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
        return std::make_optional(mExpr);
    }
    StatementWrapper::StatementWrapper(const Location loc, Expression expr) : IStatement(loc), mExpr{std::move(expr)} {}
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
    StatementWrapper *StatementWrapper::as_statement_wrapper()
    {
        return this;
    }

    std::shared_ptr<ASTNode> StatementWrapper::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        return std::make_shared<StatementWrapper>(loc, mExpr.clone(vmap));
    }
} // namespace voila::ast
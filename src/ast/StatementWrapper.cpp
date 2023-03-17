#include "ast/StatementWrapper.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp" // for ASTVisitor
#include "ast/IStatement.hpp" // for IStatement
#include <utility>            // for move
#include <vector>             // for allocator

namespace voila::ast
{
    std::string StatementWrapper::type2string() const { return "statement wrapper"; }
    bool StatementWrapper::is_statement_wrapper() const { return true; }
    StatementWrapper::StatementWrapper(const Location loc, ASTNodeVariant expr)
        : IStatement(loc), mExpr{std::move(expr)}
    {
    }
    void StatementWrapper::print(std::ostream &) const {}

    StatementWrapper *StatementWrapper::as_statement_wrapper() { return this; }

    ASTNodeVariant StatementWrapper::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        return std::make_shared<StatementWrapper>(loc, std::visit(cloneVisitor, mExpr));
    }
} // namespace voila::ast
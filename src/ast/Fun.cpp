#include "ast/Fun.hpp"
#include "ast/ASTVisitor.hpp"
#include <utility>

namespace voila::ast
{
    Fun::Fun(const Location loc, std::string fun, std::vector<Expression> args, std::vector<Statement> exprs) :
        ASTNode(loc), name{std::move(fun)}, args{std::move(args)}, body{std::move(exprs)}
    {
        // TODO: check function, deduce argument types and register function
    }
    bool Fun::is_function_definition() const
    {
        return true;
    }
    Fun *Fun::as_function_definition()
    {
        return this;
    }
    std::string Fun::type2string() const
    {
        return "function definition";
    }
    void Fun::print(std::ostream &o) const
    {
        o << name << "(";
        for (auto &arg : args)
            o << arg << ",";
        o << ")";
    }
    void Fun::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Fun::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast
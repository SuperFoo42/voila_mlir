#include "ast/Fun.hpp"

#include <utility>

namespace voila::ast
{
    Fun::Fun(std::string fun, std::vector<std::string> args, std::vector<Statement> exprs) :
        ASTNode(), name{std::move(fun)}, args{std::move(args)}, body{std::move(exprs)}
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
        for (const auto &arg : args)
        {
            o << arg << ",";
        }
        o << ")";
    }
} // namespace voila::ast
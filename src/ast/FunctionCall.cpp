#include "ast/FunctionCall.hpp"

namespace voila::ast
{
    bool FunctionCall::is_function_call() const
    {
        return true;
    }
    FunctionCall *FunctionCall::as_function_call()
    {
        return this;
    }
    FunctionCall::FunctionCall(const Location loc, std::string fun, std::vector<Expression> args) :
        IStatement(loc), fun{std::move(fun)}, args{std::move(args)}
    {
    }
    std::string FunctionCall::type2string() const
    {
        return "function call";
    }
    void FunctionCall::print(std::ostream &ostream) const
    {
        ostream << fun << "(";
        for (const auto &arg : args)
        {
            ostream << arg << ",";
        }
        ostream << ")";
    }

    void FunctionCall::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void FunctionCall::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast
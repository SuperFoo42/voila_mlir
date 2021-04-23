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
    FunctionCall::FunctionCall(std::string fun, std::vector<std::string> args) :
        IStatement(), fun{std::move(fun)}, args{std::move(args)}
    {
        // TODO: lookup function definition and check if all arguments match and have references
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
} // namespace voila::ast
#pragma once
#include "IExpression.hpp"
namespace voila::ast
{
    class Ref : IExpression
    {
      public:
        Ref(const std::string &var) : IExpression()
        {
            // TODO find reference or error
        }

        bool is_reference() const final
        {
            return true;
        }

        std::string type2string() const override
        {
            return "reference";
        }

        Ref *as_reference() final
        {
            return this;
        }

        Expression ref;
    };
} // namespace voila::ast
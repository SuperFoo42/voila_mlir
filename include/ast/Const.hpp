#pragma once
#include "IExpression.hpp"
namespace voila::ast
{
    class Const : IExpression
    {
      public:
        virtual ~Const() = default;

        bool is_const() const final
        {
            return true;
        }

        Const *as_const() final
        {
            return this;
        }

        std::string type2string() const override
        {
            return "constant";
        }
    };
} // namespace voila::ast
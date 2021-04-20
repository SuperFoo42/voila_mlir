#pragma once
#include "ASTNode.hpp"

namespace voila::ast
{
    class Arithmetic : ASTNode
    {
      public:
        virtual ~Arithmetic() = default;

        bool is_arithmetic() const final
        {
            return true;
        }

        std::string type2string() const override
        {
            return "arithmetic";
        }
    };
} // namespace voila::ast
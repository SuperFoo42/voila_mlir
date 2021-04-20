#pragma once
#include "ast/ASTNode.hpp"
#include "voila.hpp"

namespace voila::ast
{
    class ASTVisitor
    {
      public:
        void operator()(const ASTNode &) {}
        virtual void operator()(const Arithmetic &) {}
        // TODO: other operators
    };
} // namespace voila::ast
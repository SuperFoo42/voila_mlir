#pragma once
#include "Expression.hpp"
#include "IExpression.hpp"
namespace voila::ast
{
    class Ref : public IExpression
    {
      public:
        explicit Ref(const std::string &var);

        [[nodiscard]] bool is_reference() const final;

        [[nodiscard]] std::string type2string() const override;

        Ref *as_reference() final;

        void print(std::ostream &o) const final;

        // TODO: Expression ref;
    };
} // namespace voila::ast
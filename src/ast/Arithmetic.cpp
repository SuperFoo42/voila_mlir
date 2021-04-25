#include <utility>

#include "ast/Arithmetic.hpp"

namespace voila::ast
{
    std::string Arithmetic::type2string() const
    {
        return "arithmetic";
    }
    Arithmetic *Arithmetic::as_arithmetic()
    {
        return this;
    }
    bool Arithmetic::is_arithmetic() const
    {
        return true;
    }
    void Arithmetic::print(std::ostream &) const {}
    Arithmetic::Arithmetic(Expression lhs, Expression rhs) : lhs {std::move(lhs)}, rhs{std::move(rhs)} {}
} // namespace voila::ast
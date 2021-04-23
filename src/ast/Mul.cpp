#include "ast/Mul.hpp"

namespace voila::ast
{
    std::string Mul::type2string() const
    {
        return "mul";
    }
    bool Mul::is_mul() const
    {
        return true;
    }
    Mul *Mul::as_mul()
    {
        return this;
    }
    void Mul::print(std::ostream &ostream) const
    {
        ostream << "*";
    }
} // namespace voila::ast
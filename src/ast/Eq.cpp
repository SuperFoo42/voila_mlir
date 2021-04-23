#include "ast/Eq.hpp"

namespace voila::ast
{
    std::string Eq::type2string() const
    {
        return "eq";
    }
    bool Eq::is_eq() const
    {
        return true;
    }
    Eq *Eq::as_eq()
    {
        return this;
    }
    void Eq::print(std::ostream &ostream) const
    {
        ostream << "=";
    }
} // namespace voila::ast
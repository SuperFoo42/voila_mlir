#include "ast/Neq.hpp"

namespace voila::ast
{
    std::string Neq::type2string() const
    {
        return "neq";
    }
    bool Neq::is_neq() const
    {
        return true;
    }
    Neq *Neq::as_neq()
    {
        return this;
    }
    void Neq::print(std::ostream &ostream) const
    {
        ostream << "!=";
    }
} // namespace voila::ast
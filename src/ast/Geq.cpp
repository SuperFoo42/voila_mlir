#include "ast/Geq.hpp"

namespace voila::ast
{
    std::string Geq::type2string() const
    {
        return "geq";
    }
    bool Geq::is_geq() const
    {
        return true;
    }
    Geq *Geq::as_geq()
    {
        return this;
    }
    void Geq::print(std::ostream &ostream) const
    {
        ostream << ">=";
    }
    void Geq::checkArgs(Expression &lhs, Expression &rhs)
    {
        // TODO
    }
} // namespace voila::ast
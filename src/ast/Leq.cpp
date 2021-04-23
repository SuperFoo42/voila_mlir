#include "ast/Leq.hpp"

namespace voila::ast
{
    std::string Leq::type2string() const
    {
        return "leq";
    }
    bool Leq::is_leq() const
    {
        return true;
    }
    Leq *Leq::as_leq()
    {
        return this;
    }
    void Leq::print(std::ostream &ostream) const
    {
        ostream << "<=";
    }
    void Leq::checkArgs(Expression &lhs, Expression &rhs)
    {
        // TODO
    }
} // namespace voila::ast
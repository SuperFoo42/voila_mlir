#include "ast/TupleGet.hpp"

namespace voila::ast
{
    TupleGet::TupleGet(std::string exp, const intmax_t idx) : IExpression(), expr{std::move(exp)}, idx{idx}
    {
        // TODO: check expr tuple and idx in range
    }
    bool TupleGet::is_tuple_get() const
    {
        return true;
    }
    std::string TupleGet::type2string() const
    {
        return "tuple get";
    }
    void TupleGet::print(std::ostream &ostream) const
    {
        ostream << "tuple get";
    }
} // namespace voila::ast
#include "ast/TupleCreate.hpp"

namespace voila::ast
{
    bool TupleCreate::is_tuple_create() const
    {
        return true;
    }
    std::string TupleCreate::type2string() const
    {
        return "tuple create";
    }
    TupleCreate::TupleCreate(std::vector<Expression> tupleElems) : IExpression(), elems{std::move(tupleElems)} {}
    void TupleCreate::print(std::ostream &ostream) const
    {
        ostream << "create tuple";
    }
} // namespace voila::ast
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
    TupleCreate::TupleCreate(const Location loc, std::vector<Expression> tupleElems) : IExpression(loc), elems{std::move(tupleElems)} {}
    void TupleCreate::print(std::ostream &) const {}
    void TupleCreate::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void TupleCreate::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
} // namespace voila::ast
#include "ast/TupleCreate.hpp"
#include <utility>                           // for move
#include "ast/ASTVisitor.hpp"                // for ASTVisitor
#include "ast/Expression.hpp"                // for Expression
#include "ast/IExpression.hpp"               // for IExpression
#include "range/v3/algorithm/transform.hpp"  // for transform, transform_fn
#include "range/v3/functional/identity.hpp"  // for identity

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

    std::shared_ptr<ASTNode> TupleCreate::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        std::vector<Expression> clonedElems;
        ranges::transform(elems, clonedElems.begin(), [&vmap](auto &elem) { return elem.clone(vmap);});
        return std::make_shared<TupleCreate>(loc, clonedElems);
    }
} // namespace voila::ast
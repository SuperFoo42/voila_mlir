#include "ast/TupleGet.hpp"
#include "llvm/Support/FormatVariadic.h"

namespace voila::ast
{
    TupleGet::TupleGet(const Location loc, Expression exp, const intmax_t idx) : IExpression(loc), expr{std::move(exp)}, idx{idx}
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
        ostream << expr << llvm::formatv("[{0}]", idx).str();
    }
    void TupleGet::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void TupleGet::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }

    std::shared_ptr<ASTNode> TupleGet::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        return std::make_shared<TupleGet>(loc, expr.clone(vmap), idx);
    }
} // namespace voila::ast
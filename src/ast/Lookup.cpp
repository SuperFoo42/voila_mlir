#include "ast/Lookup.hpp"
#include "ast/ASTVisitor.hpp"               // for ASTVisitor
#include "ast/Expression.hpp"               // for Expression
#include "range/v3/algorithm/transform.hpp" // for transform, transform_fn
#include "range/v3/functional/identity.hpp" // for identity

namespace voila::ast
{
    bool Lookup::is_lookup() const { return true; }
    Lookup *Lookup::as_lookup() { return this; }
    std::string Lookup::type2string() const { return "hash_insert"; }
    void Lookup::print(std::ostream &) const {}
    void Lookup::visit(ASTVisitor &visitor) const { visitor(*this); }
    void Lookup::visit(ASTVisitor &visitor) { visitor(*this); }

    std::shared_ptr<ASTNode> Lookup::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap)
    {
        std::vector<Expression> clonedValues;
        ranges::transform(mValues, clonedValues.begin(), [&vmap](auto &item) { return item.clone(vmap); });
        std::vector<Expression> clonedTables;
        ranges::transform(mTables, clonedTables.begin(), [&vmap](auto &item) { return item.clone(vmap); });
        return std::make_shared<Lookup>(loc, clonedValues, clonedTables, mHashes.clone(vmap));
    }
} // namespace voila::ast
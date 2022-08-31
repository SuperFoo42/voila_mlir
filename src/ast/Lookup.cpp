#include "ast/Lookup.hpp"
#include "range/v3/algorithm.hpp"

namespace voila::ast
{
    bool Lookup::is_lookup() const
    {
        return true;
    }
    Lookup *Lookup::as_lookup()
    {
        return this;
    }
    std::string Lookup::type2string() const
    {
        return "hash_insert";
    }
    void Lookup::print(std::ostream &) const {}
    void Lookup::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Lookup::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }

    std::unique_ptr<ASTNode> Lookup::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        std::vector<Expression> clonedValues;
        ranges::transform(values, clonedValues.begin(), [&vmap](auto &item) { return item.clone(vmap); });
        std::vector<Expression> clonedTables;
        ranges::transform(tables, clonedTables.begin(), [&vmap](auto &item) { return item.clone(vmap); });
        return std::make_unique<Lookup>(loc, clonedValues, clonedTables, hashes.clone(vmap));
    }
} // namespace voila::ast
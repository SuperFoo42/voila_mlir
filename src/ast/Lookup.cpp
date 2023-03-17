#include "ast/Lookup.hpp"
#include "ast/ASTVisitor.hpp"               // for ASTVisitor
#include "range/v3/algorithm/transform.hpp" // for transform, transform_fn
#include "range/v3/functional/identity.hpp" // for identity
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"

namespace voila::ast
{
    bool Lookup::is_lookup() const { return true; }
    Lookup *Lookup::as_lookup() { return this; }
    std::string Lookup::type2string() const { return "hash_insert"; }
    void Lookup::print(std::ostream &) const {}

    ASTNodeVariant Lookup::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        std::vector<ASTNodeVariant> clonedValues;
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        for (auto &item : mValues)
        {
            clonedValues.push_back(std::visit(cloneVisitor, item));
        }
        std::vector<ASTNodeVariant> clonedTables;
        for (auto &item : mTables)
        {
            clonedTables.push_back(std::visit(cloneVisitor, item));
        }
        return std::make_shared<Lookup>(loc, clonedValues, clonedTables,std::visit(cloneVisitor, mHashes));
    }

    Lookup::Lookup(Location loc,
                   std::vector<ASTNodeVariant> values,
                   std::vector<ASTNodeVariant> tables,
                   ASTNodeVariant hashes)
        : IExpression(loc), mHashes{std::move(hashes)}, mTables{std::move(tables)}, mValues{std::move(values)}
    {
        // TODO
    }
} // namespace voila::ast
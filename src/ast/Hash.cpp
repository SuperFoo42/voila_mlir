#include "ast/Hash.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp"               // for ASTVisitor
#include "range/v3/algorithm/transform.hpp" // for transform, transform_fn
#include "range/v3/functional/identity.hpp" // for identity

namespace voila::ast
{
    [[nodiscard]] std::string Hash::type2string() const { return "hash"; }

    [[nodiscard]] bool Hash::is_hash() const { return true; }

    Hash *Hash::as_hash() { return this; }
    void Hash::print(std::ostream &) const {}

    ASTNodeVariant Hash::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        std::vector<ASTNodeVariant> clonedItems;
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};

        for (auto &item : mItems)
        {
            clonedItems.push_back(std::visit(cloneVisitor, item));
        }

        return std::make_shared<Hash>(loc, clonedItems);
    }
} // namespace voila::ast
#include "ast/Insert.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/ASTVisitor.hpp"               // for ASTVisitor
#include "range/v3/algorithm/transform.hpp" // for transform, transform_fn
#include "range/v3/functional/identity.hpp" // for identity

namespace voila::ast
{
    bool Insert::is_insert() const { return true; }
    Insert *Insert::as_insert() { return this; }
    std::string Insert::type2string() const { return "hash_insert"; }
    void Insert::print(std::ostream &) const {}

    ASTNodeVariant Insert::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
    {
        std::vector<ASTNodeVariant> clonedValues;
        auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                       [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
        for (auto &item : mValues)
        {
            clonedValues.push_back(std::visit(cloneVisitor, item));
        }

        return std::make_shared<Insert>(loc, std::visit(cloneVisitor, mKeys), clonedValues);
    }
} // namespace voila::ast
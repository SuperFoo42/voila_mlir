#include "ast/Hash.hpp"
#include "range/v3/algorithm.hpp"
namespace voila::ast
{
    [[nodiscard]] std::string Hash::type2string() const
    {
        return "hash";
    }

    [[nodiscard]] bool Hash::is_hash() const
    {
        return true;
    }

    Hash *Hash::as_hash()
    {
        return this;
    }
    void Hash::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Hash::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }
    void Hash::print(std::ostream &) const {}

    std::shared_ptr<ASTNode> Hash::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        std::vector<Expression> clonedItems;
        ranges::transform(mItems,clonedItems.begin(), [&vmap](auto &item) { return item.clone(vmap);});
        return std::make_shared<Hash>(loc, clonedItems);
    }
} // namespace voila::ast
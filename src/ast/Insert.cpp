#include "ast/Insert.hpp"
#include "range/v3/algorithm.hpp"
namespace voila::ast
{
    bool Insert::is_insert() const
    {
        return true;
    }
    Insert *Insert::as_insert()
    {
        return this;
    }
    std::string Insert::type2string() const
    {
        return "hash_insert";
    }
    void Insert::print(std::ostream &) const {

    }
    void Insert::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void Insert::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }

    std::unique_ptr<ASTNode> Insert::clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) {
        std::vector<Expression> clonedValues;
        ranges::transform(values,clonedValues.begin(), [&vmap](auto &item) { return item.clone(vmap);});
        return std::make_unique<Insert>(loc, keys.clone(vmap),clonedValues);
    }
} // namespace voila::ast
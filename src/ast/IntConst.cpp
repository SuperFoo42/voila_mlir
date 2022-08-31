#include "ast/IntConst.hpp"

namespace voila::ast
{
    bool IntConst::is_integer() const
    {
        return true;
    }
    IntConst *IntConst::as_integer()
    {
        return this;
    }
    std::string IntConst::type2string() const
    {
        return "integer";
    }
    void IntConst::print(std::ostream &ostream) const
    {
        ostream << std::to_string(val);
    }
    void IntConst::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void IntConst::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }

    std::unique_ptr<ASTNode> IntConst::clone(llvm::DenseMap<ASTNode *, ASTNode *> &) {
        return std::make_unique<IntConst>(loc, val);
    }
} // namespace voila::ast
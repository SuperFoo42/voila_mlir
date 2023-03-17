#include "ast/IntConst.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor

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
    ASTNodeVariant IntConst::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &) {
        return std::make_shared<IntConst>(loc, val);
    }
} // namespace voila::ast
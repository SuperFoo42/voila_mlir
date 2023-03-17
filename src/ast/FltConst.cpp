#include "ast/FltConst.hpp"
#include "ast/ASTVisitor.hpp"  // for ASTVisitor

namespace voila::ast
{
    bool FltConst::is_float() const
    {
        return true;
    }
    FltConst *FltConst::as_float()
    {
        return this;
    }
    std::string FltConst::type2string() const
    {
        return "float";
    }
    void FltConst::print(std::ostream &ostream) const
    {
        ostream << std::to_string(val);
    }

    ASTNodeVariant FltConst::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &) {
        return std::make_shared<FltConst>(loc, val);
    }
} // namespace voila::ast
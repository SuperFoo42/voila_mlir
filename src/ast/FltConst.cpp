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

    void FltConst::visit(ASTVisitor &visitor) const
    {
        visitor(*this);
    }
    void FltConst::visit(ASTVisitor &visitor)
    {
        visitor(*this);
    }

    std::shared_ptr<ASTNode> FltConst::clone(llvm::DenseMap<ASTNode *, ASTNode *> &) {
        return std::make_shared<FltConst>(loc, val);
    }
} // namespace voila::ast
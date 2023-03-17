#include "ast/StrConst.hpp"
#include <ostream>             // for operator<<, basic_ostream, ostream
#include "ast/ASTVisitor.hpp"  // for ASTVisitor

namespace voila::ast
{
    bool StrConst::is_string() const
    {
        return true;
    }
    std::string StrConst::type2string() const
    {
        return "string";
    }
    void StrConst::print(std::ostream &ostream) const
    {
        ostream << "\"" << val << "\"";
    }

    ASTNodeVariant StrConst::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &) {
        return std::make_shared<StrConst>(loc, val);
    }
} // namespace voila::ast
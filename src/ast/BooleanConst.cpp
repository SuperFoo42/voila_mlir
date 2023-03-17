#include "ast/BooleanConst.hpp"
#include <ios>                 // for boolalpha, noboolalpha, ostream
#include "ast/ASTVisitor.hpp"  // for ASTVisitor

namespace voila::ast
{
    bool BooleanConst::is_bool() const
    {
        return true;
    }
    BooleanConst *BooleanConst::as_bool()
    {
        return this;
    }
    std::string BooleanConst::type2string() const
    {
        return "bool";
    }
    void BooleanConst::print(std::ostream &ostream) const
    {
        ostream << std::boolalpha << val << std::noboolalpha;
    }

    ASTNodeVariant BooleanConst::clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &) {
        return std::make_shared<BooleanConst>(loc, val);
    }
} // namespace voila::ast
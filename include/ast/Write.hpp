
#pragma once

#include "Expression.hpp"      // for Expression
#include "IStatement.hpp"      // for IStatement
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string

namespace voila::ast
{
    class ASTVisitor;

    class Write : public IStatement
    {
        Expression mDest;
        Expression mStart;
        Expression mSrc;

      public:
        Write(Location loc, Expression src_col, Expression dest_col, Expression wpos);

        [[nodiscard]] bool is_write() const final;

        Write *as_write() final;

        [[nodiscard]] std::string type2string() const final;

        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor) const final;

        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        [[nodiscard]] const Expression &dest() const { return mDest; }

        [[nodiscard]] const Expression &start() const { return mStart; }

        [[nodiscard]] const Expression &src() const { return mSrc; }
    };

} // namespace voila::ast
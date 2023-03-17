#pragma once

#include "IStatement.hpp"      // for IStatement
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string

namespace voila::ast
{
    class Write : public IStatement
    {
        ASTNodeVariant mDest;
        ASTNodeVariant mStart;
        ASTNodeVariant mSrc;

      public:
        Write(Location loc, ASTNodeVariant src_col, ASTNodeVariant dest_col, ASTNodeVariant wpos);

        [[nodiscard]] bool is_write() const final;

        Write *as_write() final;

        [[nodiscard]] std::string type2string() const final;

        void print(std::ostream &ostream) const final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) override;

        [[nodiscard]] const ASTNodeVariant &dest() const;

        [[nodiscard]] const ASTNodeVariant &start() const;

        [[nodiscard]] const ASTNodeVariant &src() const;
    };
} // namespace voila::ast
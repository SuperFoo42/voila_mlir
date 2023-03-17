#pragma once

#include "IExpression.hpp"     // for IExpression
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move
#include <vector>              // for vector

namespace voila::ast
{
    class Lookup : public IExpression
    {
        ASTNodeVariant mHashes;
        std::vector<ASTNodeVariant> mTables, mValues;

      public:
        Lookup(Location loc, std::vector<ASTNodeVariant> values, std::vector<ASTNodeVariant> tables, ASTNodeVariant hashes);

        [[nodiscard]] bool is_lookup() const final;

        Lookup *as_lookup() final;

        [[nodiscard]] std::string type2string() const final;

        void print(std::ostream &ostream) const final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) override;

        [[nodiscard]] const ASTNodeVariant &hashes() const { return mHashes; }

        [[nodiscard]] const std::vector<ASTNodeVariant> &tables() const { return mTables; }

        [[nodiscard]] const std::vector<ASTNodeVariant> &values() const { return mValues; }
    };
} // namespace voila::ast
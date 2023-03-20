#pragma once

#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move
#include <vector>              // for vector
#include "range/v3/all.hpp"

namespace voila::ast
{
    class Lookup : public AbstractASTNode<Lookup>
    {
        ASTNodeVariant mHashes;
        std::vector<ASTNodeVariant> mTables, mValues;

      public:
        Lookup(Location loc,
               ranges::input_range auto &&values,
               ranges::input_range auto &&tables,
               ASTNodeVariant hashes)
            : AbstractASTNode<Lookup>(loc),
              mHashes{std::move(hashes)},
              mTables(ranges::to<std::vector>(tables)),
              mValues(ranges::to<std::vector>(values))
        {
            // TODO
        }

        [[nodiscard]] std::string type2string_impl() const { return "hash_insert"; };

        void print_impl(std::ostream &) const {};

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap);

        [[nodiscard]] const ASTNodeVariant &hashes() const { return mHashes; }

        [[nodiscard]] const std::vector<ASTNodeVariant> &tables() const { return mTables; }

        [[nodiscard]] const std::vector<ASTNodeVariant> &values() const { return mValues; }
    };
} // namespace voila::ast
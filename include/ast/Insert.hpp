#pragma once
#include "ast/ASTNode.hpp" // for ASTNode (ptr only), Location
#include "range/v3/all.hpp"
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move
#include <vector>              // for vector

namespace voila::ast
{
    class Insert : public AbstractASTNode<Insert>
    {
        ASTNodeVariant mKeys;
        std::vector<ASTNodeVariant> mValues;

      public:
        Insert(Location loc, ASTNodeVariant keys, ranges::input_range auto &&values)
            : AbstractASTNode<Insert>(loc), mKeys{std::move(keys)}, mValues(ranges::to<std::vector>(values))
        {
            // TODO
        }

        [[nodiscard]] std::string type2string_impl() const { return "hash_insert"; }
        void print_impl(std::ostream &) const {};

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap);

        [[nodiscard]] const ASTNodeVariant &keys() const { return mKeys; }

        [[nodiscard]] const std::vector<ASTNodeVariant> &values() const { return mValues; }
    };
} // namespace voila::ast
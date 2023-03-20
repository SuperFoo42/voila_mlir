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
    class Hash : public AbstractASTNode<Hash>
    {
        std::vector<ASTNodeVariant> mItems;

      public:
        Hash(const Location loc, ranges::input_range auto && items)
            : AbstractASTNode<Hash>(loc), mItems(ranges::to<std::vector>(items))
        {
        }

        [[nodiscard]] std::string type2string_impl() const { return "hash"; };

        void print_impl(std::ostream &) const {};

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap);

        [[nodiscard]] const std::vector<ASTNodeVariant> &items() const { return mItems; }
    };
} // namespace voila::ast
#pragma once
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    class Gather : public AbstractASTNode<Gather>
    {
        ASTNodeVariant mColumn, mIdxs;

      public:
        Gather(const Location loc, ASTNodeVariant lhs, ASTNodeVariant rhs)
            : AbstractASTNode(loc), mColumn{std::move(lhs)}, mIdxs{std::move(rhs)}
        {
            // TODO
        }

        [[nodiscard]] std::string type2string_impl() const { return "gather"; };
        void print_impl(std::ostream &ostream) const;

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap);

        [[nodiscard]] const ASTNodeVariant &column() const { return mColumn; }

        [[nodiscard]] const ASTNodeVariant &idxs() const { return mIdxs; }
    };
} // namespace voila::ast
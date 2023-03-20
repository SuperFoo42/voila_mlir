#pragma once
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    class Read : public AbstractASTNode<Read>
    {
        ASTNodeVariant mColumn, mIdx;

      public:
        Read(Location loc, ASTNodeVariant lhs, ASTNodeVariant rhs)
            : AbstractASTNode<Read>(loc), mColumn{std::move(lhs)}, mIdx{std::move(rhs)}
        {
            // TODO
        }

        [[nodiscard]] std::string type2string_impl() const { return "read"; };
        void print_impl(std::ostream &) const {};

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap);

        [[nodiscard]] const ASTNodeVariant &column() const { return mColumn; }

        [[nodiscard]] const ASTNodeVariant &idx() const { return mIdx; }
    };
} // namespace voila::ast
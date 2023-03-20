#pragma once

#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for make_shared, shared_ptr
#include <optional>            // for optional, nullopt
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    template <class Aggr> class Aggregation : public AbstractASTNode<Aggr>
    {
        ASTNodeVariant mSrc;
        ASTNodeVariant mGroups;

      public:
        Aggregation(const Location loc, ASTNodeVariant col)
            : AbstractASTNode<Aggr>(loc), mSrc{std::move(col)}, mGroups()
        {
        }
        Aggregation(const Location loc, ASTNodeVariant col, ASTNodeVariant groups)
            : AbstractASTNode<Aggr>(loc), mSrc{std::move(col)}, mGroups(std::move(groups))
        {
        }

        void print_impl(std::ostream &) const {}

        [[nodiscard]] const ASTNodeVariant &src() const { return mSrc; }

        [[nodiscard]] const ASTNodeVariant &groups() const { return mGroups; }

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap)
        {
            auto cloneVisitor =overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                           [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};;
            return std::make_shared<Aggr>(this->loc, std::visit(cloneVisitor, mSrc),
                                          std::visit(cloneVisitor, mGroups));
        }
    };
} // namespace voila::ast
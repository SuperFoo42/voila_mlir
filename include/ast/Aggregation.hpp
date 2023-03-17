#pragma once

#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "ast/IExpression.hpp" // for IExpression
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for make_shared, shared_ptr
#include <optional>            // for optional, nullopt
#include <string>              // for string
#include <utility>             // for move

namespace voila::ast
{
    class Aggregation : public IExpression
    {
        ASTNodeVariant mSrc;
        ASTNodeVariant mGroups;

      public:
        Aggregation(const Location loc, ASTNodeVariant col) : IExpression(loc), mSrc{std::move(col)}, mGroups() {}
        Aggregation(const Location loc, ASTNodeVariant col, ASTNodeVariant groups)
            : IExpression(loc), mSrc{std::move(col)}, mGroups(std::move(groups))
        {
        }
        ~Aggregation() override = default;

        [[nodiscard]] bool is_aggr() const final;

        Aggregation *as_aggr() final;

        [[nodiscard]] std::string type2string() const override;

        void print(std::ostream &) const final {}

        [[nodiscard]] const ASTNodeVariant &src() const { return mSrc; }

        [[nodiscard]] const ASTNodeVariant &groups() const
        {
                return mGroups;
        }

      protected:
        template <class T>
        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap)
            requires std::is_base_of_v<Aggregation, T>
        {
            auto cloneVisitor = overloaded{[&vmap](auto &e) -> ASTNodeVariant { return e->clone(vmap); },
                                           [](std::monostate &) -> ASTNodeVariant { throw std::logic_error(""); }};
            return std::make_shared<T>(loc, std::visit(cloneVisitor, mSrc), std::visit(cloneVisitor, mGroups));
        }
    };
} // namespace voila::ast
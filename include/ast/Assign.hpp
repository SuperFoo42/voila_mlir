#pragma once
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "range/v3/all.hpp"    // for all_of, all_of_fn
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <optional>            // for optional
#include <string>              // for string
#include <vector>              // for vector

namespace voila::ast
{
    class Assign : public AbstractASTNode<Assign>
    {
        ASTNodeVariant pred;
        std::vector<ASTNodeVariant> mDdests;
        ASTNodeVariant mExpr;

      public:
        Assign(Location loc, ranges::input_range auto &&dests, ASTNodeVariant expr)
            : AbstractASTNode<Assign>(loc), pred{}, mDdests(ranges::to<std::vector>(dests)), mExpr{std::move(expr)}
        {
            assert(ranges::all_of(this->mDdests,
                                  [](auto &dest) -> auto
                                  {
                                      return std::visit(overloaded{[](auto) { return false; },
                                                                   [](const std::shared_ptr<Variable> &)
                                                                   { return true; },
                                                                   [](const std::shared_ptr<Ref> &) { return true; }},
                                                        dest);
                                  }));
        }

        [[nodiscard]] std::string type2string_impl() const;

        void print_impl(std::ostream &ostream) const;

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap);

        void set_predicate(ASTNodeVariant expression);

        std::optional<ASTNodeVariant> get_predicate();

        [[nodiscard]] const std::vector<ASTNodeVariant> &dests() const { return mDdests; }

        [[nodiscard]] const ASTNodeVariant &expr() const { return mExpr; };
    };

} // namespace voila::ast
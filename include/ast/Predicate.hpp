#pragma once
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string

namespace voila::ast
{
    /**
     * @brief Meta node to wrap expressions into predicates
     * @deprecated
     */
    class Predicate : public AbstractASTNode<Predicate>
    {
        ASTNodeVariant mExpr;

      public:
        explicit Predicate(Location loc, ASTNodeVariant expr) : AbstractASTNode<Predicate>(loc), mExpr(std::move(expr))
        {
        }

        [[nodiscard]] std::string type2string_impl() const { return "predicate"; }
        void print_impl(std::ostream &) const {}

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap);

        [[nodiscard]] const ASTNodeVariant &expr() const { return mExpr; }
    };

} // namespace voila::ast

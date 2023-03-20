#pragma once

#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <optional>            // for optional
#include <string>              // for string

namespace voila::ast
{
    /**
     * @brief Meta node to wrap expressions into statements
     *
     */
    class StatementWrapper : public AbstractASTNode<StatementWrapper>
    {
        ASTNodeVariant mExpr;

      public:
        explicit StatementWrapper(Location loc, ASTNodeVariant expr)
            : AbstractASTNode<StatementWrapper>(loc), mExpr{std::move(expr)}
        {
        }

        [[nodiscard]] std::string type2string_impl() const { return "statement wrapper"; }

        void print_impl(std::ostream &) const {};

        ASTNodeVariant clone_impl(std::unordered_map<ASTNodeVariant, ASTNodeVariant> &vmap);

        [[nodiscard]] const ASTNodeVariant &expr() const { return mExpr; }
    };


} // namespace voila::ast
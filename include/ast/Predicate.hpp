#pragma once
#include "IExpression.hpp"     // for IExpression
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
    class Predicate : public IExpression
    {
        ASTNodeVariant mExpr;

      public:
        explicit Predicate(Location loc, ASTNodeVariant expr);

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_predicate() const final;

        Predicate *as_predicate() final;

        void print(std::ostream &ostream) const final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) override;

        [[nodiscard]] const ASTNodeVariant &expr() const { return mExpr; }
    };

} // namespace voila::ast

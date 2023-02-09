#pragma once
#include <cstdint>              // for intmax_t
#include <iosfwd>               // for ostream
#include <memory>               // for shared_ptr
#include <string>               // for string
#include "Expression.hpp"       // for Expression
#include "IExpression.hpp"      // for IExpression
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap

namespace voila::ast
{
    class ASTVisitor;
    //TODO
    /**
     * @deprecated ?
     */
    class TupleGet : public IExpression
    {
      public:
        TupleGet(Location loc, Expression exp, intmax_t idx);

        [[nodiscard]] bool is_tuple_get() const final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        Expression expr;
        const std::intmax_t idx;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;
    };
} // namespace voila::ast
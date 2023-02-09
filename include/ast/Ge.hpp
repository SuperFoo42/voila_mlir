#pragma once
#include <memory>               // for shared_ptr
#include <string>               // for string
#include <utility>              // for move
#include "Comparison.hpp"       // for Comparison
#include "Expression.hpp"       // for Expression
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap

namespace voila::ast
{
    class ASTVisitor;

    class Ge : public Comparison
    {
      public:
        Ge(Location loc, Expression lhs, Expression rhs) : Comparison(loc, std::move(lhs), std::move(rhs))
        {
            // TODO
        }

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_ge() const final;

        Ge *as_ge() final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) final
        {
            return Comparison::clone<Ge>(vmap);
        }
    };
} // namespace voila::ast
#pragma once
#include <iosfwd>               // for ostream
#include <memory>               // for shared_ptr
#include <string>               // for string
#include <utility>              // for move
#include <vector>               // for vector
#include "Expression.hpp"       // for Expression
#include "IExpression.hpp"      // for IExpression
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap

namespace voila::ast
{
    class ASTVisitor;

    class Insert : public IExpression
    {
        Expression mKeys;
        std::vector<Expression> mValues;

      public:
        Insert(Location loc, Expression keys, std::vector<Expression> values) :
            IExpression(loc), mKeys{std::move(keys)}, mValues{std::move(values)}
        {
            // TODO
        }

        [[nodiscard]] bool is_insert() const final;

        Insert *as_insert() final;

        [[nodiscard]] std::string type2string() const final;
        void print(std::ostream &ostream) const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        [[nodiscard]] const Expression &keys() const {
            return mKeys;
        }

        [[nodiscard]] const std::vector<Expression> &values() const {
            return mValues;
        }
    };
} // namespace voila::ast
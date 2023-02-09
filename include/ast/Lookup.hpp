#pragma once

#include "Expression.hpp"      // for Expression
#include "IExpression.hpp"     // for IExpression
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <string>              // for string
#include <utility>             // for move
#include <vector>              // for vector

namespace voila::ast
{
    class ASTVisitor;

    class Lookup : public IExpression
    {
        Expression mHashes;
        std::vector<Expression> mTables, mValues;

      public:
        Lookup(Location loc, std::vector<Expression> values, std::vector<Expression> tables, Expression hashes)
            : IExpression(loc), mHashes{std::move(hashes)}, mTables{std::move(tables)}, mValues{std::move(values)}
        {
            // TODO
        }

        [[nodiscard]] bool is_lookup() const final;

        Lookup *as_lookup() final;

        [[nodiscard]] std::string type2string() const final;

        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor) const final;

        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        [[nodiscard]] const Expression &hashes() const { return mHashes; }

        [[nodiscard]] const std::vector<Expression> &tables() const { return mTables; }

        [[nodiscard]] const std::vector<Expression> &values() const { return mValues; }
    };
} // namespace voila::ast
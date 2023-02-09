#pragma once
#include "Expression.hpp"      // for Expression
#include "IStatement.hpp"      // for IStatement
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <ast/Statement.hpp>   // for Statement
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr
#include <optional>            // for optional
#include <string>              // for string
#include <vector>              // for vector

namespace voila::ast
{
    class ASTVisitor;

    class Assign : public IStatement
    {
        std::optional<Expression> pred;
        std::vector<Expression> mDdests;
        Statement mExpr;

      public:
        Assign(Location loc, std::vector<Expression> dests, Statement expr);

        [[nodiscard]] bool is_assignment() const final;

        Assign *as_assignment() final;

        [[nodiscard]] std::string type2string() const final;

        void set_predicate(Expression expression) final;
        std::optional<Expression> get_predicate() final;

        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor) final;
        void visit(ASTVisitor &visitor) const final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        const std::vector<Expression> &dests() const { return mDdests; }

        const Statement &expr() const { return mExpr; };
    };

} // namespace voila::ast
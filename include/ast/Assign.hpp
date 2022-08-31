#pragma once
#include "ASTVisitor.hpp"
#include "Expression.hpp"
#include "IStatement.hpp"
#include "Predicate.hpp"
#include "Statement.hpp"

#include <ast/Statement.hpp>
#include <cassert>
#include <optional>
#include <range/v3/all.hpp>
#include <utility>
#include <vector>

namespace voila::ast
{
    class Assign : public IStatement
    {
        std::optional<Expression> pred;

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

        std::unique_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        const std::vector<Expression> dests;
        const Statement expr;
    };

} // namespace voila::ast
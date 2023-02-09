#pragma once

#include <iosfwd>               // for ostream
#include <memory>               // for shared_ptr
#include <optional>             // for optional
#include <string>               // for string
#include "Expression.hpp"       // for Expression
#include "IStatement.hpp"       // for IStatement
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap

namespace voila::ast {
    class ASTVisitor;

    /**
     * @brief Meta node to wrap expressions into statements
     *
     */
    class StatementWrapper : public IStatement {
        Expression mExpr;

    public:
        explicit StatementWrapper(Location loc, Expression expr);

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_statement_wrapper() const final;

        [[nodiscard]] StatementWrapper *as_statement_wrapper() final;

        std::optional<Expression> as_expression() final;

        void print(std::ostream &ostream) const final;

        void visit(ASTVisitor &visitor) const final;

        void visit(ASTVisitor &visitor) final;

        std::shared_ptr<ASTNode> clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) override;

        const Expression &expr() const
        {
            return mExpr;
        }
    };

} // namespace voila::ast
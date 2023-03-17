#pragma once

#include <iosfwd>               // for ostream
#include <memory>               // for shared_ptr
#include <optional>             // for optional
#include <string>               // for string
#include "IStatement.hpp"       // for IStatement
#include "ast/ASTNode.hpp"      // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h"  // for DenseMap

namespace voila::ast {
    /**
     * @brief Meta node to wrap expressions into statements
     *
     */
    class StatementWrapper : public IStatement {
        ASTNodeVariant mExpr;

    public:
        explicit StatementWrapper(Location loc, ASTNodeVariant expr);

        [[nodiscard]] std::string type2string() const final;

        [[nodiscard]] bool is_statement_wrapper() const final;

        [[nodiscard]] StatementWrapper *as_statement_wrapper() final;

        void print(std::ostream &ostream) const final;

        ASTNodeVariant clone(llvm::DenseMap<AbstractASTNode *, AbstractASTNode *> &vmap) override;

        [[nodiscard]] const ASTNodeVariant &expr() const
        {
            return mExpr;
        }
    };

} // namespace voila::ast
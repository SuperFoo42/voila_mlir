#pragma once
#include "ast/ASTNode.hpp"     // for ASTNode (ptr only), Location
#include "llvm/ADT/DenseMap.h" // for DenseMap
#include <iosfwd>              // for ostream
#include <memory>              // for shared_ptr, make_shared
#include <optional>            // for optional
#include <utility>             // for move, forward

namespace voila::ast
{
    class ASTVisitor;
    class Assign;
    class Emit;
    class Expression;
    class FunctionCall;
    class IStatement;
    class Loop;
    class StatementWrapper;
    class Write;

    class Statement
    {
        std::shared_ptr<IStatement> mImpl;
        explicit Statement(std::shared_ptr<IStatement> impl) : mImpl{std::move(impl)} {}

      public:
        Statement() = default;
        Statement(Statement &) = default;
        Statement(const Statement &) = default;
        Statement(Statement &&) = default;

        Statement &operator=(const Statement &) = default;

        template <typename StmtImpl, typename... Args>
            requires std::is_base_of_v<IStatement, StmtImpl>
        static Statement make(Args &&...args)
        {
            return Statement(std::make_shared<StmtImpl>(std::forward<Args>(args)...));
        }

        friend std::ostream &operator<<(std::ostream &out, const Statement &t);

        [[nodiscard]] bool is_stmt() const;

        [[nodiscard]] bool is_loop() const;

        [[nodiscard]] bool is_assignment() const;

        [[nodiscard]] bool is_emit() const;

        [[nodiscard]] bool is_function_call() const;

        [[nodiscard]] bool is_write() const;

        // type conversions
        [[nodiscard]] IStatement *as_stmt() const;

        [[nodiscard]] Loop *as_loop() const;

        [[nodiscard]] Assign *as_assignment() const;

        [[nodiscard]] Emit *as_emit() const;

        [[nodiscard]] FunctionCall *as_function_call() const;

        [[nodiscard]] Write *as_write() const;

        [[nodiscard]] bool is_statement_wrapper() const;
        [[nodiscard]] StatementWrapper *as_statement_wrapper() const;

        [[nodiscard]] std::optional<Expression> as_expression() const;

        void visit(ASTVisitor &visitor);
        void visit(ASTVisitor &visitor) const;

        void set_predicate(Expression expr);

        /*TODO: do we need this?
            size_t get_table_column_ref(std::string &tbl_col) const;
            size_t get_table_column_ref(std::string &tbl, std::string &col) const;
            size_t get_table_ref(std::string &tbl) const;

            bool has_result() const;

            //virtual std::string type2string() const = 0;
            */
        Location get_location();
        std::optional<Expression> get_predicate();

        Statement clone(llvm::DenseMap<ASTNode *, ASTNode *> &vmap) const;
    };
} // namespace voila::ast
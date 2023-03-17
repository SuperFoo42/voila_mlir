#pragma once

#include "ASTNode.hpp"

#include <optional>

namespace voila::ast
{
    class Expression;
    class Write;
    class FunctionCall;
    class Assign;
    class Emit;
    class Loop;
    class StatementWrapper;

    class IStatement : public AbstractASTNode
    {
      public:
        explicit IStatement(const Location loc) : AbstractASTNode(loc){};
        ~IStatement() override = default;

        virtual void set_predicate(ASTNodeVariant);
        virtual std::optional<ASTNodeVariant> get_predicate();

        // type checks
        [[nodiscard]] bool is_stmt() const final;

        [[nodiscard]] virtual bool is_loop() const;

        [[nodiscard]] virtual bool is_assignment() const;

        [[nodiscard]] virtual bool is_emit() const;

        [[nodiscard]] virtual bool is_function_call() const;

        [[nodiscard]] virtual bool is_write() const;

        [[nodiscard]] virtual bool is_statement_wrapper() const;

        [[nodiscard]] virtual StatementWrapper *as_statement_wrapper();

        // type conversions
        virtual IStatement *as_stmt();

        virtual Loop *as_loop();

        virtual Assign *as_assignment();

        virtual Emit *as_emit();

        virtual FunctionCall *as_function_call();

        virtual Write *as_write();
    };
} // namespace voila::ast

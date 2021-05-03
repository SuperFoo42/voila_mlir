#pragma once

#include "ASTNode.hpp"
#include "Expression.hpp"

#include <optional>

namespace voila::ast
{
    class Aggregation;
    class AggrSum;
    class AggrCnt;
    class AggrMin;
    class AggrMax;
    class AggrAvg;
    class Write;
    class Scatter;
    class FunctionCall;
    class Assign;
    class Emit;
    class Loop;
    class StatementWrapper;
    class Predicate;

    class IStatement : public ASTNode
    {
      public:
        explicit IStatement(const Location loc) :
            ASTNode(loc) {};
        ~IStatement() override = default;

        virtual void set_predicate(Expression);
        virtual std::optional<Expression> get_predicate();

        // type checks
        [[nodiscard]] bool is_stmt() const final;

        [[nodiscard]] virtual bool is_aggr() const;

        [[nodiscard]] virtual bool is_loop() const;

        [[nodiscard]] virtual bool is_assignment() const;

        [[nodiscard]] virtual bool is_emit() const;

        [[nodiscard]] virtual bool is_function_call() const;

        [[nodiscard]] virtual bool is_scatter() const;

        [[nodiscard]] virtual bool is_write() const;

        [[nodiscard]] virtual bool is_aggr_sum() const;

        [[nodiscard]] virtual bool is_aggr_cnt() const;

        [[nodiscard]] virtual bool is_aggr_min() const;

        [[nodiscard]] virtual bool is_aggr_max() const;

        [[nodiscard]] virtual bool is_aggr_avg() const;

        [[nodiscard]] virtual bool is_statement_wrapper() const;

        // type conversions
        virtual IStatement *as_stmt();

        virtual Aggregation *as_aggr();

        virtual Loop *as_loop();

        virtual Assign *as_assignment();

        virtual Emit *as_emit();

        virtual FunctionCall *as_function_call();

        virtual Scatter *as_scatter();

        virtual Write *as_write();

        virtual AggrSum *as_aggr_sum();

        virtual AggrCnt *as_aggr_cnt();

        virtual AggrMin *as_aggr_min();

        virtual AggrMax *as_aggr_max();

        virtual AggrAvg *as_aggr_avg();

        virtual std::optional<Expression> as_expression();
    };
} // namespace voila::ast

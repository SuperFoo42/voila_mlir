#pragma once
#include "IStatement.hpp"

#include <algorithm>
#include <concepts>
#include <memory>
#include <utility>

namespace voila::ast
{
    class ASTVisitor;

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

        template<typename StmtImpl, typename... Args>
        requires std::is_base_of_v<IStatement, StmtImpl>
        static Statement make(Args &&...args)
        {
            return Statement(std::make_shared<StmtImpl>(std::forward<Args>(args)...));
        }

        friend std::ostream &operator<<(std::ostream &out, const Statement &t);

        [[nodiscard]] bool is_stmt() const;

        [[nodiscard]] bool is_aggr() const;

        [[nodiscard]] bool is_loop() const;

        [[nodiscard]] bool is_assignment() const;

        [[nodiscard]] bool is_emit() const;

        [[nodiscard]] bool is_function_call() const;

        [[nodiscard]] bool is_scatter() const;

        [[nodiscard]] bool is_write() const;

        [[nodiscard]] bool is_aggr_sum() const;

        [[nodiscard]] bool is_aggr_cnt() const;

        [[nodiscard]] bool is_aggr_min() const;

        [[nodiscard]] bool is_aggr_max() const;

        [[nodiscard]] bool is_aggr_avg() const;
        // type conversions
        IStatement *as_stmt();

        Aggregation *as_aggr();

        Loop *as_loop();

        Assign *as_assignment();

        Emit *as_emit();

        FunctionCall *as_function_call();

        Scatter *as_scatter();

        Write *as_write();

        AggrSum *as_aggr_sum();

        AggrCnt *as_aggr_cnt();

        AggrMin *as_aggr_min();

        AggrMax *as_aggr_max();

        AggrAvg *as_aggr_avg();

        [[nodiscard]] bool is_statement_wrapper() const;

        std::optional<Expression> as_expression();

        void visit(ASTVisitor &visitor);
        void visit(ASTVisitor &visitor) const;

        void predicate(Expression expr);

        /*TODO: do we need this?
            size_t get_table_column_ref(std::string &tbl_col) const;
            size_t get_table_column_ref(std::string &tbl, std::string &col) const;
            size_t get_table_ref(std::string &tbl) const;

            bool has_result() const;

            //virtual std::string type2string() const = 0;
            */
        Location get_location();
    };
} // namespace voila::ast
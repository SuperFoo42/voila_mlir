#include "ast/Statement.hpp"

namespace voila::ast
{
    std::ostream &operator<<(std::ostream &out, const Statement &t)
    {
        t.mImpl->print(out);
        return out;
    }

    bool Statement::is_stmt() const
    {
        return true;
    }
    bool Statement::is_aggr() const
    {
        return mImpl->is_aggr();
    }
    bool Statement::is_loop() const
    {
        return mImpl->is_loop();
    }
    bool Statement::is_assignment() const
    {
        return mImpl->is_assignment();
    }
    bool Statement::is_emit() const
    {
        return mImpl->is_emit();
    }
    bool Statement::is_function_call() const
    {
        return mImpl->is_function_call();
    }
    bool Statement::is_scatter() const
    {
        return mImpl->is_scatter();
    }
    bool Statement::is_write() const
    {
        return mImpl->is_write();
    }
    bool Statement::is_aggr_sum() const
    {
        return mImpl->is_aggr_sum();
    }
    bool Statement::is_aggr_cnt() const
    {
        return mImpl->is_aggr_cnt();
    }
    bool Statement::is_aggr_min() const
    {
        return mImpl->is_aggr_min();
    }
    bool Statement::is_aggr_max() const
    {
        return mImpl->is_aggr_max();
    }
    bool Statement::is_aggr_avg() const
    {
        return mImpl->is_aggr_avg();
    }
    IStatement *Statement::as_stmt()
    {
        return mImpl->as_stmt();
    }
    Aggregation *Statement::as_aggr()
    {
        return mImpl->as_aggr();
    }
    Loop *Statement::as_loop()
    {
        return mImpl->as_loop();
    }
    Assign *Statement::as_assignment()
    {
        return mImpl->as_assignment();
    }
    Emit *Statement::as_emit()
    {
        return mImpl->as_emit();
    }
    Scatter *Statement::as_scatter()
    {
        return mImpl->as_scatter();
    }
    Write *Statement::as_write()
    {
        return mImpl->as_write();
    }
    AggrSum *Statement::as_aggr_sum()
    {
        return mImpl->as_aggr_sum();
    }
    AggrCnt *Statement::as_aggr_cnt()
    {
        return mImpl->as_aggr_cnt();
    }
    AggrMin *Statement::as_aggr_min()
    {
        return mImpl->as_aggr_min();
    }
    AggrMax *Statement::as_aggr_max()
    {
        return mImpl->as_aggr_max();
    }
    AggrAvg *Statement::as_aggr_avg()
    {
        return mImpl->as_aggr_avg();
    }
    bool Statement::is_statement_wrapper() const
    {
        return mImpl->is_statement_wrapper();
    }
    std::optional<Expression> Statement::as_expression()
    {
        return mImpl->as_expression();
    }
    void Statement::visit(ASTVisitor &visitor)
    {
        mImpl->visit(visitor);
    }
    void Statement::predicate(Expression expr)
    {
        mImpl->predicate(std::move(expr));
    }
} // namespace voila::ast
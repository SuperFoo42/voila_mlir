#include "ast/IStatement.hpp"

namespace voila::ast
{
    void IStatement::predicate(Expression)
    {
        throw PredicationUnsupportedException(fmt::format("Predication not supported for: {}.", type2string()));
    }
    bool IStatement::is_stmt() const
    {
        return true;
    }
    bool IStatement::is_aggr() const
    {
        return false;
    }
    bool IStatement::is_loop() const
    {
        return false;
    }
    bool IStatement::is_assignment() const
    {
        return false;
    }
    bool IStatement::is_emit() const
    {
        return false;
    }
    bool IStatement::is_write() const
    {
        return false;
    }
    bool IStatement::is_scatter() const
    {
        return false;
    }
    bool IStatement::is_function_call() const
    {
        return false;
    }
    bool IStatement::is_aggr_sum() const
    {
        return false;
    }
    bool IStatement::is_aggr_cnt() const
    {
        return false;
    }
    bool IStatement::is_aggr_min() const
    {
        return false;
    }
    bool IStatement::is_aggr_max() const
    {
        return false;
    }
    bool IStatement::is_aggr_avg() const
    {
        return false;
    }
    bool IStatement::is_statement_wrapper() const
    {
        return false;
    }
    IStatement *IStatement::as_stmt()
    {
        return this;
    }
    Aggregation *IStatement::as_aggr()
    {
        return nullptr;
    }
    Loop *IStatement::as_loop()
    {
        return nullptr;
    }
    Assign *IStatement::as_assignment()
    {
        return nullptr;
    }
    Emit *IStatement::as_emit()
    {
        return nullptr;
    }
    FunctionCall *IStatement::as_function_call()
    {
        return nullptr;
    }
    Scatter *IStatement::as_scatter()
    {
        return nullptr;
    }
    Write *IStatement::as_write()
    {
        return nullptr;
    }
    AggrSum *IStatement::as_aggr_sum()
    {
        return nullptr;
    }
    AggrCnt *IStatement::as_aggr_cnt()
    {
        return nullptr;
    }
    AggrMin *IStatement::as_aggr_min()
    {
        return nullptr;
    }
    AggrMax *IStatement::as_aggr_max()
    {
        return nullptr;
    }
    AggrAvg *IStatement::as_aggr_avg()
    {
        return nullptr;
    }
    std::optional<Expression> IStatement::as_expression()
    {
        return std::nullopt;
    }
} // namespace voila::ast
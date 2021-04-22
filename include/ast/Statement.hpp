#pragma once
#include "IStatement.hpp"

#include <algorithm>
#include <concepts>
#include <memory>

namespace voila::ast
{
    class ASTVisitor;

    class Statement
    {
        std::shared_ptr<IStatement> mImpl;
        Statement(std::shared_ptr<IStatement> impl) : mImpl{impl} {}

      public:
        template<typename StmtImpl, typename... Args>
        requires std::derived_from<StmtImpl, IStatement> static Statement make(Args &&...args)
        {
            return Statement(std::shared_ptr<StmtImpl>(new StmtImpl(std::forward<Args>(args)...)));
        }

        friend std::ostream &operator<<(std::ostream &out, const Statement &t)
        {
            t.mImpl->print(out);
            return out;
        }

        bool is_stmt() const
        {
            return true;
        }

        bool is_aggr() const
        {
            return mImpl->is_aggr();
        }

        bool is_loop() const
        {
            return mImpl->is_loop();
        }

        bool is_assignment() const
        {
            return mImpl->is_assignment();
        }

        bool is_emit() const
        {
            return mImpl->is_emit();
        }

        bool is_function_call() const
        {
            return mImpl->is_function_call();
        }

        bool is_scatter() const
        {
            return mImpl->is_scatter();
        }

        bool is_write() const
        {
            return mImpl->is_write();
        }

        bool is_aggr_sum() const
        {
            return mImpl->is_aggr_sum();
        }

        bool is_aggr_cnt() const
        {
            return mImpl->is_aggr_cnt();
        }

        bool is_aggr_min() const
        {
            return mImpl->is_aggr_min();
        }

        bool is_aggr_max() const
        {
            return mImpl->is_aggr_max();
        }

        bool is_aggr_avg() const
        {
            return mImpl->is_aggr_avg();
        }
//type conversions
        IStatement * as_stmt()
        {
            return mImpl->as_stmt();
        }

        Aggregation * as_aggr()
        {
            return mImpl->as_aggr();
        }

        Loop * as_loop()
        {
            return mImpl->as_loop();
        }

        Assign * as_assignment()
        {
            return mImpl->as_assignment();
        }

        Emit * as_emit()
        {
            return mImpl->as_emit();
        }

        FunctionCall * as_function_call()
        {
            return mImpl->as_function_call();
        }

        Scatter * as_scatter()
        {
            return mImpl->as_scatter();
        }

        Write * as_write()
        {
            return mImpl->as_write();
        }

        AggrSum * as_aggr_sum()
        {
            return mImpl->as_aggr_sum();
        }

        AggrCnt * as_aggr_cnt()
        {
            return mImpl->as_aggr_cnt();
        }

        AggrMin * as_aggr_min()
        {
            return mImpl->as_aggr_min();
        }

        AggrMax * as_aggr_max()
        {
            return mImpl->as_aggr_max();
        }

        AggrAvg * as_aggr_avg()
        {
            return mImpl->as_aggr_avg();
        }

        void visit(ASTVisitor &visitor)
        {
            mImpl->visit(visitor);
        }
        /*TODO: do we need this?
            size_t get_table_column_ref(std::string &tbl_col) const;
            size_t get_table_column_ref(std::string &tbl, std::string &col) const;
            size_t get_table_ref(std::string &tbl) const;

            bool has_result() const;

            //virtual std::string type2string() const = 0;
            */
    };
} // namespace voila::ast
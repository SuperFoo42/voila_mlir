#pragma once

#include "ASTNode.hpp"

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

    class IStatement : public ASTNode
    {
      public:
        virtual ~IStatement() = default;
        using ASTNode::print;
        using ASTNode::type2string;
        using ASTNode::visit;

        // type checks
        bool is_stmt() const final
        {
            return true;
        }

        virtual bool is_aggr() const
        {
            return false;
        }

        virtual bool is_loop() const
        {
            return false;
        }

        virtual bool is_assignment() const
        {
            return false;
        }

        virtual bool is_emit() const
        {
            return false;
        }

        virtual bool is_function_call() const
        {
            return false;
        }

        virtual bool is_scatter() const
        {
            return false;
        }

        virtual bool is_write() const
        {
            return false;
        }

        virtual bool is_aggr_sum() const
        {
            return false;
        }

        virtual bool is_aggr_cnt() const
        {
            return false;
        }

        virtual bool is_aggr_min() const
        {
            return false;
        }

        virtual bool is_aggr_max() const
        {
            return false;
        }

        virtual bool is_aggr_avg() const
        {
            return false;
        }
//type conversions
        virtual IStatement * as_stmt()
        {
            return this;
        }

        virtual Aggregation * as_aggr()
        {
            return nullptr;
        }

        virtual Loop * as_loop()
        {
            return nullptr;
        }

        virtual Assign * as_assignment()
        {
            return nullptr;
        }

        virtual Emit * as_emit()
        {
            return nullptr;
        }

        virtual FunctionCall * as_function_call()
        {
            return nullptr;
        }

        virtual Scatter * as_scatter()
        {
            return nullptr;
        }

        virtual Write * as_write()
        {
            return nullptr;
        }

        virtual AggrSum * as_aggr_sum()
        {
            return nullptr;
        }

        virtual AggrCnt * as_aggr_cnt()
        {
            return nullptr;
        }

        virtual AggrMin * as_aggr_min()
        {
            return nullptr;
        }

        virtual AggrMax * as_aggr_max()
        {
            return nullptr;
        }

        virtual AggrAvg * as_aggr_avg()
        {
            return nullptr;
        }
    };
} // namespace voila::ast

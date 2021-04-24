#pragma once
#include "ASTVisitor.hpp"
#include "Fun.hpp"
#include "FunctionCall.hpp"
#include "IntConst.hpp"
#include "FltConst.hpp"
#include "StrConst.hpp"
#include "BooleanConst.hpp"

#include <iostream>
#include "fmt/core.h"

namespace voila::ast
{
    class DotVisualizer : public ASTVisitor
    {
        Fun &to_dot;
        std::ostream *os;
        size_t nodeID;

      public:
        explicit DotVisualizer(Fun &start) : to_dot{start}, os{nullptr}, nodeID{0} {
        }
        void operator()(const AggrSum &sum) final;
        void operator()(const AggrCnt &cnt) final;
        void operator()(const AggrMin &min) final;
        void operator()(const AggrMax &max) final;
        void operator()(const AggrAvg &avg) final;
        void operator()(const Write &write) final;
        void operator()(const Scatter &scatter) final;
        void operator()(const FunctionCall &call) final;
        void operator()(const Assign &assign) final;
        void operator()(const Emit &emit) final;
        void operator()(const Loop &loop) final;
        void operator()(const StatementWrapper &wrapper) final;
        void operator()(const Add &add) final;
        void operator()(const Sub &sub) final;
        void operator()(const Mul &mul) final;
        void operator()(const Div &div) final;
        void operator()(const Mod &mod) final;
        void operator()(const Eq &eq) final;
        void operator()(const Neq &neq) final;
        void operator()(const Le &le) final;
        void operator()(const Ge &ge) final;
        void operator()(const Leq &leq) final;
        void operator()(const Geq &geq) final;
        void operator()(const And &anAnd) final;
        void operator()(const Or &anOr) final;
        void operator()(const Not &aNot) final;
        void operator()(const IntConst &aConst) final;
        void operator()(const BooleanConst &aConst) final;
        void operator()(const FltConst &aConst) final;
        void operator()(const StrConst &aConst) final;
        void operator()(const Read &read) final;
        void operator()(const Gather &gather) final;
        void operator()(const Ref &param) final;
        void operator()(const TupleGet &get) final;
        void operator()(const TupleCreate &create) final;
        void operator()(const Fun &create) final;

        std::ostream &operator<<(std::ostream &out);
    };
} // namespace voila::ast
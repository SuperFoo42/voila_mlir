#pragma once
#include "ASTVisitor.hpp"
#include "ASTNodes.hpp"
#include <fmt/core.h>
#include <iostream>

namespace voila::ast
{
    class DotVisualizer : public ASTVisitor
    {
      protected:
        Fun &to_dot;
        std::ostream *os;
        size_t nodeID;

      public:
        explicit DotVisualizer(Fun &start) : to_dot{start}, os{nullptr}, nodeID{0} {}
        void operator()(AggrSum &sum) final;
        void operator()(AggrCnt &cnt) final;
        void operator()(AggrMin &min) final;
        void operator()(AggrMax &max) final;
        void operator()(AggrAvg &avg) final;
        void operator()(Write &write) final;
        void operator()(Scatter &scatter) final;
        void operator()(FunctionCall &call) final;
        void operator()(Assign &assign) final;
        void operator()(Emit &emit) final;
        void operator()(Loop &loop) final;
        void operator()(StatementWrapper &wrapper) final;
        void operator()(Add &add) final;
        void operator()(Sub &sub) final;
        void operator()(Mul &mul) final;
        void operator()(Div &div) final;
        void operator()(Mod &mod) final;
        void operator()(Eq &eq) final;
        void operator()(Neq &neq) final;
        void operator()(Le &le) final;
        void operator()(Ge &ge) final;
        void operator()(Leq &leq) final;
        void operator()(Geq &geq) final;
        void operator()(And &anAnd) final;
        void operator()(Or &anOr) final;
        void operator()(Not &aNot) final;
        void operator()(IntConst &aConst) final;
        void operator()(BooleanConst &aConst) final;
        void operator()(FltConst &aConst) final;
        void operator()(StrConst &aConst) final;
        void operator()(Read &read) final;
        void operator()(Gather &gather) final;
        void operator()(Ref &param) final;
        void operator()(TupleGet &get) final;
        void operator()(TupleCreate &create) final;
        void operator()(Main &create) final;
        void operator()(Fun &create) final;
        void operator()(Selection &create) final;

        friend std::ostream &operator<<(std::ostream &out, DotVisualizer &t);

      private:
        void printVertex(ASTNode &node);
    };
} // namespace voila::ast
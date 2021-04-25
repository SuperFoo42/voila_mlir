#pragma once
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
    class Selection;
    class Const;
    class Add;
    class Arithmetic;
    class Sub;
    class Mul;
    class Div;
    class Mod;
    class Comparison;
    class Eq;
    class Neq;
    class Le;
    class Ge;
    class Leq;
    class Geq;
    class And;
    class Or;
    class Not;
    class Logical;
    class IntConst;
    class BooleanConst;
    class FltConst;
    class StrConst;
    class Read;
    class Gather;
    class Ref;
    class TupleGet;
    class TupleCreate;
    class ASTNode;
    class Fun;
    class Main;

    class ASTVisitor
    {
      public:
        virtual ~ASTVisitor() = default;
        virtual void operator()(const ASTNode&){}
        virtual void operator()(const Aggregation &) {}
        virtual void operator()(const AggrSum &) {}
        virtual void operator()(const AggrCnt &) {}
        virtual void operator()(const AggrMin &) {}
        virtual void operator()(const AggrMax &) {}
        virtual void operator()(const AggrAvg &) {}
        virtual void operator()(const Write &) {}
        virtual void operator()(const Scatter &) {}
        virtual void operator()(const FunctionCall &) {}
        virtual void operator()(const Assign &) {}
        virtual void operator()(const Emit &) {}
        virtual void operator()(const Loop &) {}
        virtual void operator()(const StatementWrapper &) {}
        virtual void operator()(const Const &) {}
        virtual void operator()(const Add &) {}
        virtual void operator()(const Arithmetic &) {}
        virtual void operator()(const Sub &) {}
        virtual void operator()(const Mul &) {}
        virtual void operator()(const Div &) {}
        virtual void operator()(const Mod &) {}
        virtual void operator()(const Eq &) {}
        virtual void operator()(const Neq &) {}
        virtual void operator()(const Le &) {}
        virtual void operator()(const Ge &) {}
        virtual void operator()(const Leq &) {}
        virtual void operator()(const Geq &) {}
        virtual void operator()(const And &) {}
        virtual void operator()(const Or &) {}
        virtual void operator()(const Not &) {}
        virtual void operator()(const Logical &) {}
        virtual void operator()(const IntConst &) {}
        virtual void operator()(const BooleanConst &) {}
        virtual void operator()(const FltConst &) {}
        virtual void operator()(const StrConst &) {}
        virtual void operator()(const Read &) {}
        virtual void operator()(const Gather &) {}
        virtual void operator()(const Ref &) {}
        virtual void operator()(const TupleGet &) {}
        virtual void operator()(const TupleCreate &) {}
        virtual void operator()(const Fun &) {}
        virtual void operator()(const Main &) {}
        virtual void operator()(const Selection &) {}

        virtual void operator()(ASTNode&){}
        virtual void operator()(Aggregation &) {}
        virtual void operator()(AggrSum &) {}
        virtual void operator()(AggrCnt &) {}
        virtual void operator()(AggrMin &) {}
        virtual void operator()(AggrMax &) {}
        virtual void operator()(AggrAvg &) {}
        virtual void operator()(Write &) {}
        virtual void operator()(Scatter &) {}
        virtual void operator()(FunctionCall &) {}
        virtual void operator()(Assign &) {}
        virtual void operator()(Emit &) {}
        virtual void operator()(Loop &) {}
        virtual void operator()(StatementWrapper &) {}
        virtual void operator()(Const &) {}
        virtual void operator()(Add &) {}
        virtual void operator()(Arithmetic &) {}
        virtual void operator()(Sub &) {}
        virtual void operator()(Mul &) {}
        virtual void operator()(Div &) {}
        virtual void operator()(Mod &) {}
        virtual void operator()(Eq &) {}
        virtual void operator()(Neq &) {}
        virtual void operator()(Le &) {}
        virtual void operator()(Ge &) {}
        virtual void operator()(Leq &) {}
        virtual void operator()(Geq &) {}
        virtual void operator()(And &) {}
        virtual void operator()(Or &) {}
        virtual void operator()(Not &) {}
        virtual void operator()(Logical &) {}
        virtual void operator()(IntConst &) {}
        virtual void operator()(BooleanConst &) {}
        virtual void operator()(FltConst &) {}
        virtual void operator()(StrConst &) {}
        virtual void operator()(Read &) {}
        virtual void operator()(Gather &) {}
        virtual void operator()(Ref &) {}
        virtual void operator()(TupleGet &) {}
        virtual void operator()(TupleCreate &) {}
        virtual void operator()(Fun &) {}
        virtual void operator()(Main &) {}
        virtual void operator()(Selection &) {}

    };
} // namespace voila::ast

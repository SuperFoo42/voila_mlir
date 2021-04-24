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
        void operator()(const ASTNode &){};
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
    };
} // namespace voila::ast
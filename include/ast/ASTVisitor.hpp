#pragma once

#include <stdexcept>

namespace voila::ast {
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

    class Comparison;

    class Variable;

    class Predicate;

    class Hash;

    class Lookup;

    class Insert;

    class ASTVisitor {
    public:
        virtual ~ASTVisitor() = default;

        virtual void operator()(const ASTNode &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const Aggregation &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const AggrSum &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const AggrCnt &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const AggrMin &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const AggrMax &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const AggrAvg &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const Write &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const Scatter &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(FunctionCall &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const FunctionCall &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const Assign &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const Emit &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        virtual void operator()(const Loop &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        virtual void operator()(const StatementWrapper &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const Const &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const Add &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        virtual void operator()(const Arithmetic &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const Sub &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        virtual void operator()(const Mul &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        virtual void operator()(const Div &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        virtual void operator()(const Mod &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        virtual void operator()(const Eq &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        virtual void operator()(const Neq &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        virtual void operator()(const Le &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        virtual void operator()(const Ge &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        virtual void operator()(const Leq &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        virtual void operator()(const Geq &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        virtual void operator()(const And &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        virtual void operator()(const Or &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        virtual void operator()(const Not &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        virtual void operator()(const Logical &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const Comparison &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const IntConst &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const BooleanConst &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const FltConst &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const StrConst &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const Read &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        virtual void operator()(const Gather &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const Ref &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        virtual void operator()(const TupleGet &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const TupleCreate &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const Fun &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        virtual void operator()(const Main &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        virtual void operator()(const Selection &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const Variable &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const Predicate &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const Hash &) { throw std::logic_error("Pure ASTNodes are not allowed to be visited"); }

        //TODO: insert in visitors
        virtual void operator()(const Lookup &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual void operator()(const Insert &) {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }
    };
} // namespace voila::ast

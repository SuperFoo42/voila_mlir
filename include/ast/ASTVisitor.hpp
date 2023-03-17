#pragma once

#include <stdexcept>

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

    class AbstractASTNode;

    class Fun;

    class Main;

    class Comparison;

    class Variable;

    class Predicate;

    class Hash;

    class Lookup;

    class Insert;

    template <typename T = void> class ASTVisitor
    {
      public:
        using result_type = T;
        virtual ~ASTVisitor() = default;

        virtual T operator()(std::shared_ptr<Aggregation>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<AggrSum>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<AggrCnt>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<AggrMin>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<AggrMax>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<AggrAvg>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Write>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Scatter>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<FunctionCall>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Assign>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Emit>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Loop>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<StatementWrapper>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Const>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Add>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Arithmetic>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Sub>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Mul>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Div>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Mod>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Eq>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Neq>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Le>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Ge>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Leq>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Geq>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<And>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Or>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Not>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Logical>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Comparison>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<IntConst>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<BooleanConst>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<FltConst>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<StrConst>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Read>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Gather>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Ref>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Fun>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Main>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Selection>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Variable>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Predicate>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Hash>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Lookup>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::shared_ptr<Insert>)
        {
            throw std::logic_error("Pure ASTNodes are not allowed to be visited");
        }

        virtual T operator()(std::monostate)
        {
            throw std::logic_error("No ASTNode");
        }
    };
} // namespace voila::ast

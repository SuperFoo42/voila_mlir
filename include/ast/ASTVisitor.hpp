#pragma once

#include <stdexcept>
#include <type_traits>

namespace voila::ast
{
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
    class Sub;
    class Mul;
    class Div;
    class Mod;
    class Eq;
    class Neq;
    class Le;
    class Ge;
    class Leq;
    class Geq;
    class And;
    class Or;
    class Not;
    class IntConst;
    class BooleanConst;
    class FltConst;
    class StrConst;
    class Read;
    class Gather;
    class Ref;
    class Fun;
    class Main;
    class Variable;
    class Predicate;
    class Hash;
    class Lookup;
    class Insert;
    class Load;

    template <class Derived, typename R> class ASTVisitor
    {

      public:
        using return_type = R;
        virtual ~ASTVisitor() = default;

        return_type operator()(std::shared_ptr<AggrSum> aggrSum)
        {
            return visit(aggrSum);
        }

        return_type operator()(std::shared_ptr<AggrCnt> aggrCnt)
        {
            return visit(aggrCnt);
        }

        return_type operator()(std::shared_ptr<AggrMin> aggrMin)
        {
            return visit(aggrMin);
        }

        return_type operator()(std::shared_ptr<AggrMax> aggrMax)
        {
            return visit(aggrMax);
        }

        return_type operator()(std::shared_ptr<AggrAvg> aggrAvg)
        {
            return visit(aggrAvg);
        }

        return_type operator()(std::shared_ptr<Write> write) { return visit(write); }

        return_type operator()(std::shared_ptr<Scatter> scatter)
        {
            return visit(scatter);
        }

        return_type operator()(std::shared_ptr<FunctionCall> functionCall)
        {
            return visit(functionCall);
        }

        return_type operator()(std::shared_ptr<Assign> assign)
        {
            return visit(assign);
        }

        return_type operator()(std::shared_ptr<Emit> emit) { return visit(emit); }

        return_type operator()(std::shared_ptr<Loop> loop) { return visit(loop); }

        return_type operator()(std::shared_ptr<StatementWrapper> statementWrapper)
        {
            return visit(statementWrapper);
        }

        return_type operator()(std::shared_ptr<Add> add) { return visit(add); }

        return_type operator()(std::shared_ptr<Sub> sub) { return visit(sub); }

        return_type operator()(std::shared_ptr<Mul> mul) { return visit(mul); }

        return_type operator()(std::shared_ptr<Div> div) { return visit(div); }

        return_type operator()(std::shared_ptr<Mod> mod) { return visit(mod); }

        return_type operator()(std::shared_ptr<Eq> eq) { return visit(eq); }

        return_type operator()(std::shared_ptr<Neq> neq) { return visit(neq); }

        return_type operator()(std::shared_ptr<Le> le) { return visit(le); }

        return_type operator()(std::shared_ptr<Ge> ge) { return visit(ge); }

        return_type operator()(std::shared_ptr<Leq> leq) { return visit(leq); }

        return_type operator()(std::shared_ptr<Geq> geq) { return visit(geq); }

        return_type operator()(std::shared_ptr<And> anAnd) { return visit(anAnd); }

        return_type operator()(std::shared_ptr<Or> anOr) { return visit(anOr); }

        return_type operator()(std::shared_ptr<Not> aNot) { return visit(aNot); }

        return_type operator()(std::shared_ptr<IntConst> intConst)
        {
            return visit(intConst);
        }

        return_type operator()(std::shared_ptr<BooleanConst> booleanConst)
        {
            return visit(booleanConst);
        }

        return_type operator()(std::shared_ptr<FltConst> fltConst)
        {
            return visit(fltConst);
        }

        return_type operator()(std::shared_ptr<StrConst> strConst)
        {
            return visit(strConst);
        }

        return_type operator()(std::shared_ptr<Read> read) { return visit(read); }

        return_type operator()(std::shared_ptr<Gather> gather)
        {
            return visit(gather);
        }

        return_type operator()(std::shared_ptr<Ref> ref) { return visit(ref); }

        return_type operator()(std::shared_ptr<Fun> fun) { return visit(fun); }

        return_type operator()(std::shared_ptr<Main> main) { return visit(main); }

        return_type operator()(std::shared_ptr<Selection> selection)
        {
            return visit(selection);
        }

        return_type operator()(std::shared_ptr<Variable> variable)
        {
            return visit(variable);
        }

        return_type operator()(std::shared_ptr<Predicate> predicate)
        {
            return visit(predicate);
        }

        return_type operator()(std::shared_ptr<Hash> hash) { return visit(hash); }

        return_type operator()(std::shared_ptr<Lookup> lookup) { return visit(lookup); }

        return_type operator()(std::shared_ptr<Insert> insert) { return visit(insert); }

        return_type operator()(std::shared_ptr<Load> load) { return visit(load); }

        return_type operator()(std::monostate m) { return visit(m); }

      private:
        template <class T> return_type constexpr visit(T m) { return static_cast<Derived *>(this)->visit_impl(m); }
    };

} // namespace voila::ast

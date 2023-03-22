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

    template <class Derived, typename R> class ASTVisitor
    {
      public:
        using return_type = R;
        virtual ~ASTVisitor() = default;

        return_type operator()(std::shared_ptr<AggrSum> aggrSum)
        {
            return static_cast<Derived *>(this)->visit_impl(aggrSum);
        }

        return_type operator()(std::shared_ptr<AggrCnt> aggrCnt)
        {
            return static_cast<Derived *>(this)->visit_impl(aggrCnt);
        }

        return_type operator()(std::shared_ptr<AggrMin> aggrMin)
        {
            return static_cast<Derived *>(this)->visit_impl(aggrMin);
        }

        return_type operator()(std::shared_ptr<AggrMax> aggrMax)
        {
            return static_cast<Derived *>(this)->visit_impl(aggrMax);
        }

        return_type operator()(std::shared_ptr<AggrAvg> aggrAvg)
        {
            return static_cast<Derived *>(this)->visit_impl(aggrAvg);
        }

        return_type operator()(std::shared_ptr<Write> write) { return static_cast<Derived *>(this)->visit_impl(write); }

        return_type operator()(std::shared_ptr<Scatter> scatter)
        {
            return static_cast<Derived *>(this)->visit_impl(scatter);
        }

        return_type operator()(std::shared_ptr<FunctionCall> functionCall)
        {
            return static_cast<Derived *>(this)->visit_impl(functionCall);
        }

        return_type operator()(std::shared_ptr<Assign> assign)
        {
            return static_cast<Derived *>(this)->visit_impl(assign);
        }

        return_type operator()(std::shared_ptr<Emit> emit) { return static_cast<Derived *>(this)->visit_impl(emit); }

        return_type operator()(std::shared_ptr<Loop> loop) { return static_cast<Derived *>(this)->visit_impl(loop); }

        return_type operator()(std::shared_ptr<StatementWrapper> statementWrapper)
        {
            return static_cast<Derived *>(this)->visit_impl(statementWrapper);
        }

        return_type operator()(std::shared_ptr<Add> add) { return static_cast<Derived *>(this)->visit_impl(add); }

        return_type operator()(std::shared_ptr<Sub> sub) { return static_cast<Derived *>(this)->visit_impl(sub); }

        return_type operator()(std::shared_ptr<Mul> mul) { return static_cast<Derived *>(this)->visit_impl(mul); }

        return_type operator()(std::shared_ptr<Div> div) { return static_cast<Derived *>(this)->visit_impl(div); }

        return_type operator()(std::shared_ptr<Mod> mod) { return static_cast<Derived *>(this)->visit_impl(mod); }

        return_type operator()(std::shared_ptr<Eq> eq) { return static_cast<Derived *>(this)->visit_impl(eq); }

        return_type operator()(std::shared_ptr<Neq> neq) { return static_cast<Derived *>(this)->visit_impl(neq); }

        return_type operator()(std::shared_ptr<Le> le) { return static_cast<Derived *>(this)->visit_impl(le); }

        return_type operator()(std::shared_ptr<Ge> ge) { return static_cast<Derived *>(this)->visit_impl(ge); }

        return_type operator()(std::shared_ptr<Leq> leq) { return static_cast<Derived *>(this)->visit_impl(leq); }

        return_type operator()(std::shared_ptr<Geq> geq) { return static_cast<Derived *>(this)->visit_impl(geq); }

        return_type operator()(std::shared_ptr<And> anAnd) { return static_cast<Derived *>(this)->visit_impl(anAnd); }

        return_type operator()(std::shared_ptr<Or> anOr) { return static_cast<Derived *>(this)->visit_impl(anOr); }

        return_type operator()(std::shared_ptr<Not> aNot) { return static_cast<Derived *>(this)->visit_impl(aNot); }

        return_type operator()(std::shared_ptr<IntConst> intConst)
        {
            return static_cast<Derived *>(this)->visit_impl(intConst);
        }

        return_type operator()(std::shared_ptr<BooleanConst> booleanConst)
        {
            return static_cast<Derived *>(this)->visit_impl(booleanConst);
        }

        return_type operator()(std::shared_ptr<FltConst> fltConst)
        {
            return static_cast<Derived *>(this)->visit_impl(fltConst);
        }

        return_type operator()(std::shared_ptr<StrConst> strConst)
        {
            return static_cast<Derived *>(this)->visit_impl(strConst);
        }

        return_type operator()(std::shared_ptr<Read> read) { return static_cast<Derived *>(this)->visit_impl(read); }

        return_type operator()(std::shared_ptr<Gather> gather)
        {
            return static_cast<Derived *>(this)->visit_impl(gather);
        }

        return_type operator()(std::shared_ptr<Ref> ref) { return static_cast<Derived *>(this)->visit_impl(ref); }

        return_type operator()(std::shared_ptr<Fun> fun) { return static_cast<Derived *>(this)->visit_impl(fun); }

        return_type operator()(std::shared_ptr<Main> main) { return static_cast<Derived *>(this)->visit_impl(main); }

        return_type operator()(std::shared_ptr<Selection> selection)
        {
            return static_cast<Derived *>(this)->visit_impl(selection);
        }

        return_type operator()(std::shared_ptr<Variable> variable)
        {
            return static_cast<Derived *>(this)->visit_impl(variable);
        }

        return_type operator()(std::shared_ptr<Predicate> predicate)
        {
            return static_cast<Derived *>(this)->visit_impl(predicate);
        }

        return_type operator()(std::shared_ptr<Hash> hash) { return static_cast<Derived *>(this)->visit_impl(hash); }

        return_type operator()(std::shared_ptr<Lookup> lookup)
        {
            return static_cast<Derived *>(this)->visit_impl(lookup);
        }

        return_type operator()(std::shared_ptr<Insert> insert)
        {
            return static_cast<Derived *>(this)->visit_impl(insert);
        }

        return_type operator()(std::monostate m) { return static_cast<Derived *>(this)->visit_impl(m); }
    };
} // namespace voila::ast

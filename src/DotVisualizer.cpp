#include "DotVisualizer.hpp"
#include <vector>                         // for vector
#include "ast/Add.hpp"                    // for Add
#include "ast/AggrAvg.hpp"                // for AggrAvg
#include "ast/AggrCnt.hpp"                // for AggrCnt
#include "ast/AggrMax.hpp"                // for AggrMax
#include "ast/AggrMin.hpp"                // for AggrMin
#include "ast/AggrSum.hpp"                // for AggrSum
#include "ast/And.hpp"                    // for And
#include "ast/Assign.hpp"                 // for Assign
#include "ast/BooleanConst.hpp"           // for BooleanConst
#include "ast/Div.hpp"                    // for Div
#include "ast/Emit.hpp"                   // for Emit
#include "ast/Eq.hpp"                     // for Eq
#include "ast/Expression.hpp"             // for Expression
#include "ast/FltConst.hpp"               // for FltConst
#include "ast/Fun.hpp"                    // for Fun
#include "ast/FunctionCall.hpp"           // for FunctionCall
#include "ast/Gather.hpp"                 // for Gather
#include "ast/Ge.hpp"                     // for Ge
#include "ast/Geq.hpp"                    // for Geq
#include "ast/IntConst.hpp"               // for IntConst
#include "ast/Le.hpp"                     // for Le
#include "ast/Leq.hpp"                    // for Leq
#include "ast/Loop.hpp"                   // for Loop
#include "ast/Main.hpp"                   // for Main
#include "ast/Mod.hpp"                    // for Mod
#include "ast/Mul.hpp"                    // for Mul
#include "ast/Neq.hpp"                    // for Neq
#include "ast/Not.hpp"                    // for Not
#include "ast/Or.hpp"                     // for Or
#include "ast/Read.hpp"                   // for Read
#include "ast/Ref.hpp"                    // for Ref
#include "ast/Scatter.hpp"                // for Scatter
#include "ast/Selection.hpp"              // for Selection
#include "ast/Statement.hpp"              // for Statement
#include "ast/StatementWrapper.hpp"       // for StatementWrapper
#include "ast/StrConst.hpp"               // for StrConst
#include "ast/Sub.hpp"                    // for Sub
#include "ast/TupleCreate.hpp"            // for TupleCreate
#include "ast/TupleGet.hpp"               // for TupleGet
#include "ast/Variable.hpp"               // for Variable
#include "ast/Write.hpp"                  // for Write
#include "llvm/Support/FormatVariadic.h"  // for formatv, formatv_object

namespace voila::ast
{
    void DotVisualizer::operator()(const AggrSum &sum)
    {
        const auto id = nodeID;
        printVertex(sum);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        sum.src().visit(*this);
    }

    void DotVisualizer::operator()(const AggrCnt &cnt)
    {
        const auto id = nodeID;
        printVertex(cnt);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        cnt.src().visit(*this);
    }

    void DotVisualizer::operator()(const AggrMin &min)
    {
        const auto id = nodeID;
        printVertex(min);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        min.src().visit(*this);
    }

    void DotVisualizer::operator()(const AggrMax &max)
    {
        const auto id = nodeID;
        printVertex(max);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        max.src().visit(*this);
    }

    void DotVisualizer::operator()(const AggrAvg &avg)
    {
        const auto id = nodeID;
        printVertex(avg);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        avg.src().visit(*this);
    }

    void DotVisualizer::operator()(const Write &write)
    {
        const auto id = nodeID;
        printVertex(write);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        write.start().visit(*this);
    }

    void DotVisualizer::operator()(const Scatter &scatter)
    {
        const auto id = nodeID;
        printVertex(scatter);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        scatter.src().visit(*this);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        scatter.idxs().visit(*this);
    }

    void DotVisualizer::operator()(const FunctionCall &call)
    {
        printVertex(call);
    }

    void DotVisualizer::operator()(const Assign &assign)
    {
        const auto id = nodeID;
        printVertex(assign);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        for (auto dest : assign.dests())
        {
            dest.visit(*this);
        }
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        assign.expr().visit(*this);
    }

    void DotVisualizer::operator()(const Emit &emit)
    {
        const auto id = nodeID;
        printVertex(emit);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        for (auto expr : emit.exprs())
            expr.visit(*this);
    }

    void DotVisualizer::operator()(const Loop &loop)
    {
        const auto id = nodeID;
        printVertex<false>(loop);

        for (const auto &stmt : loop.stmts())
        {
            *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
            stmt.visit(*this);
        }

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        loop.pred().visit(*this);
    }

    void DotVisualizer::operator()(const Add &add)
    {
        const auto id = nodeID;
        printVertex(add);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        add.lhs().visit(*this);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        add.rhs().visit(*this);
    }

    void DotVisualizer::operator()(const Sub &sub)
    {
        const auto id = nodeID;
        printVertex(sub);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        sub.lhs().visit(*this);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        sub.rhs().visit(*this);
    }

    void DotVisualizer::operator()(const Mul &mul)
    {
        const auto id = nodeID;
        printVertex(mul);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        mul.lhs().visit(*this);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        mul.rhs().visit(*this);
    }

    void DotVisualizer::operator()(const Div &div)
    {
        const auto id = nodeID;
        printVertex(div);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        div.lhs().visit(*this);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        div.rhs().visit(*this);
    }

    void DotVisualizer::operator()(const Mod &mod)
    {
        const auto id = nodeID;
        printVertex(mod);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        mod.lhs().visit(*this);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        mod.rhs().visit(*this);
    }

    void DotVisualizer::operator()(const Eq &eq)
    {
        const auto id = nodeID;
        printVertex(eq);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        eq.lhs().visit(*this);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        eq.rhs().visit(*this);
    }

    void DotVisualizer::operator()(const Neq &neq)
    {
        const auto id = nodeID;
        printVertex(neq);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        neq.lhs().visit(*this);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        neq.rhs().visit(*this);
    }

    void DotVisualizer::operator()(const Le &le)
    {
        const auto id = nodeID;
        printVertex(le);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        le.lhs().visit(*this);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        le.rhs().visit(*this);
    }

    void DotVisualizer::operator()(const Ge &ge)
    {
        const auto id = nodeID;
        printVertex(ge);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        ge.lhs().visit(*this);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        ge.rhs().visit(*this);
    }

    void DotVisualizer::operator()(const Leq &leq)
    {
        const auto id = nodeID;
        printVertex(leq);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        leq.lhs().visit(*this);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        leq.rhs().visit(*this);
    }

    void DotVisualizer::operator()(const Geq &geq)
    {
        const auto id = nodeID;
        printVertex(geq);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        geq.lhs().visit(*this);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        geq.rhs().visit(*this);
    }

    void DotVisualizer::operator()(const And &anAnd)
    {
        const auto id = nodeID;
        printVertex(anAnd);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        anAnd.lhs().visit(*this);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        anAnd.rhs().visit(*this);
    }

    void DotVisualizer::operator()(const Or &anOr)
    {
        const auto id = nodeID;
        printVertex(anOr);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        anOr.lhs().visit(*this);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        anOr.rhs().visit(*this);
    }

    void DotVisualizer::operator()(const Not &aNot)
    {
        const auto id = nodeID;
        printVertex(aNot);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        aNot.param().visit(*this);
    }

    void DotVisualizer::operator()(const IntConst &intConst)
    {
        printVertex(intConst);
    }

    void DotVisualizer::operator()(const BooleanConst &boolConst)
    {
        printVertex(boolConst);
    }

    void DotVisualizer::operator()(const FltConst &fltConst)
    {
        printVertex(fltConst);
    }

    void DotVisualizer::operator()(const StrConst &strConst)
    {
        printVertex(strConst);
    }

    void DotVisualizer::operator()(const Read &read)
    {
        const auto id = nodeID;
        printVertex(read);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        read.column().visit(*this);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        read.idx().visit(*this);
    }

    void DotVisualizer::operator()(const Gather &gather)
    {
        const auto id = nodeID;
        printVertex(gather);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        gather.column().visit(*this);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        gather.idxs().visit(*this);
    }

    void DotVisualizer::operator()(const Ref &param)
    {
        printVertex(param);
    }

    void DotVisualizer::operator()(const TupleGet &get)
    {
        printVertex(get);
    }

    void DotVisualizer::operator()(const TupleCreate &create)
    {
        const auto id = nodeID;
        printVertex(create);

        for (const auto &elem : create.elems)
        {
            *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
            elem.visit(*this);
        }
    }

    void DotVisualizer::operator()(const Selection &create)
    {
        const auto id = nodeID;
        printVertex(create);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        create.param().visit(*this);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        create.pred().visit(*this);
    }

    void DotVisualizer::operator()(const StatementWrapper &wrapper)
    {
        const auto id = nodeID;
        printVertex<false>(wrapper);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        wrapper.expr().visit(*this);
    }

    void DotVisualizer::operator()(const Fun &fun)
    {
        const auto id = nodeID;
        printVertex(fun);

        for (const auto &stmt : fun.body())
        {
            *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
            stmt.visit(*this);
        }
    }

    void DotVisualizer::operator()(const Main &fun)
    {
        const auto id = nodeID;
        printVertex(fun);

        for (const auto &stmt : fun.body())
        {
            *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
            stmt.visit(*this);
        }
    }

    void DotVisualizer::operator()(const Variable &var)
    {
        printVertex(var);
    }

    std::ostream &operator<<(std::ostream &out, DotVisualizer &t)
    {
        out << "digraph " + t.to_dot.name() + "{\n";
        t.os = &out;
        t.nodeID = 0;
        t.to_dot.visit(t);
        t.os = nullptr;
        out << "}" << std::endl;
        return out;
    }
} // namespace voila::ast

#include "DotVisualizer.hpp"
namespace voila::ast
{
    void DotVisualizer::operator()(const AggrSum &sum)
    {
        const auto id = nodeID;
        printVertex(sum);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        sum.src.visit(*this);
    }

    void DotVisualizer::operator()(const AggrCnt &cnt)
    {
        const auto id = nodeID;
        printVertex(cnt);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        cnt.src.visit(*this);
    }

    void DotVisualizer::operator()(const AggrMin &min)
    {
        const auto id = nodeID;
        printVertex(min);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        min.src.visit(*this);
    }

    void DotVisualizer::operator()(const AggrMax &max)
    {
        const auto id = nodeID;
        printVertex(max);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        max.src.visit(*this);
    }

    void DotVisualizer::operator()(const AggrAvg &avg)
    {
        const auto id = nodeID;
        printVertex(avg);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        avg.src.visit(*this);
    }

    void DotVisualizer::operator()(const Write &write)
    {
        const auto id = nodeID;
        printVertex(write);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        write.start.visit(*this);
    }

    void DotVisualizer::operator()(const Scatter &scatter)
    {
        const auto id = nodeID;
        printVertex(scatter);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        scatter.src.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        scatter.idxs.visit(*this);
    }

    void DotVisualizer::operator()(const FunctionCall &call)
    {
        printVertex(call);
    }

    void DotVisualizer::operator()(const Assign &assign)
    {
        const auto id = nodeID;
        printVertex(assign);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        for (auto dest : assign.dests)
        {
            dest.visit(*this);
        }
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        assign.expr.visit(*this);
    }

    void DotVisualizer::operator()(const Emit &emit)
    {
        const auto id = nodeID;
        printVertex(emit);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        emit.expr.visit(*this);
    }

    void DotVisualizer::operator()(const Loop &loop)
    {
        const auto id = nodeID;
        printVertex<false>(loop);

        for (const auto &stmt : loop.stms)
        {
            *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
            stmt.visit(*this);
        }

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        loop.pred.visit(*this);
    }

    void DotVisualizer::operator()(const Add &add)
    {
        const auto id = nodeID;
        printVertex(add);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        add.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        add.rhs.visit(*this);
    }

    void DotVisualizer::operator()(const Sub &sub)
    {
        const auto id = nodeID;
        printVertex(sub);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        sub.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        sub.rhs.visit(*this);
    }

    void DotVisualizer::operator()(const Mul &mul)
    {
        const auto id = nodeID;
        printVertex(mul);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        mul.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        mul.rhs.visit(*this);
    }

    void DotVisualizer::operator()(const Div &div)
    {
        const auto id = nodeID;
        printVertex(div);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        div.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        div.rhs.visit(*this);
    }

    void DotVisualizer::operator()(const Mod &mod)
    {
        const auto id = nodeID;
        printVertex(mod);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        mod.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        mod.rhs.visit(*this);
    }

    void DotVisualizer::operator()(const Eq &eq)
    {
        const auto id = nodeID;
        printVertex(eq);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        eq.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        eq.rhs.visit(*this);
    }

    void DotVisualizer::operator()(const Neq &neq)
    {
        const auto id = nodeID;
        printVertex(neq);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        neq.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        neq.rhs.visit(*this);
    }

    void DotVisualizer::operator()(const Le &le)
    {
        const auto id = nodeID;
        printVertex(le);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        le.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        le.rhs.visit(*this);
    }

    void DotVisualizer::operator()(const Ge &ge)
    {
        const auto id = nodeID;
        printVertex(ge);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        ge.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        ge.rhs.visit(*this);
    }

    void DotVisualizer::operator()(const Leq &leq)
    {
        const auto id = nodeID;
        printVertex(leq);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        leq.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        leq.rhs.visit(*this);
    }

    void DotVisualizer::operator()(const Geq &geq)
    {
        const auto id = nodeID;
        printVertex(geq);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        geq.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        geq.rhs.visit(*this);
    }

    void DotVisualizer::operator()(const And &anAnd)
    {
        const auto id = nodeID;
        printVertex(anAnd);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        anAnd.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        anAnd.rhs.visit(*this);
    }

    void DotVisualizer::operator()(const Or &anOr)
    {
        const auto id = nodeID;
        printVertex(anOr);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        anOr.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        anOr.rhs.visit(*this);
    }

    void DotVisualizer::operator()(const Not &aNot)
    {
        const auto id = nodeID;
        printVertex(aNot);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        aNot.param.visit(*this);
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

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        read.column.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        read.idx.visit(*this);
    }

    void DotVisualizer::operator()(const Gather &gather)
    {
        const auto id = nodeID;
        printVertex(gather);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        gather.column.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        gather.idxs.visit(*this);
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
            *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
            elem.visit(*this);
        }
    }

    void DotVisualizer::operator()(const Selection &create)
    {
        const auto id = nodeID;
        printVertex(create);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        create.param.visit(*this);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        create.pred.visit(*this);
    }

    void DotVisualizer::operator()(const StatementWrapper &wrapper)
    {
        const auto id = nodeID;
        printVertex<false>(wrapper);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        wrapper.expr.visit(*this);
    }

    void DotVisualizer::operator()(const Fun &fun)
    {
        const auto id = nodeID;
        printVertex(fun);

        for (const auto &stmt : fun.body)
        {
            *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
            stmt.visit(*this);
        }
    }

    void DotVisualizer::operator()(const Main &fun)
    {
        const auto id = nodeID;
        printVertex(fun);

        for (const auto &stmt : fun.body)
        {
            *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
            stmt.visit(*this);
        }
    }

    void DotVisualizer::operator()(const Variable &var)
    {
        printVertex(var);
    }

    std::ostream &operator<<(std::ostream &out, DotVisualizer &t)
    {
        out << "digraph " + t.to_dot.name + "{\n";
        t.os = &out;
        t.nodeID = 0;
        t.to_dot.visit(t);
        t.os = nullptr;
        out << "}" << std::endl;
        return out;
    }
} // namespace voila::ast

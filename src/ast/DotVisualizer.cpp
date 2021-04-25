#include "ast/DotVisualizer.hpp"
namespace voila::ast
{
    void DotVisualizer::operator()(AggrSum &sum)
    {
        const auto id = nodeID;
        printVertex(sum);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        sum.src.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        sum.idxs.visit(*this);
    }

    void DotVisualizer::operator()(AggrCnt &cnt)
    {
        const auto id = nodeID;
        printVertex(cnt);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        cnt.src.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        cnt.idxs.visit(*this);
    }

    void DotVisualizer::operator()(AggrMin &min)
    {
        const auto id = nodeID;
        printVertex(min);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        min.src.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        min.idxs.visit(*this);
    }

    void DotVisualizer::operator()(AggrMax &max)
    {
        const auto id = nodeID;
        printVertex(max);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        max.src.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        max.idxs.visit(*this);
    }

    void DotVisualizer::operator()(AggrAvg &avg)
    {
        const auto id = nodeID;
        printVertex(avg);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        avg.src.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        avg.idxs.visit(*this);
    }

    void DotVisualizer::operator()(Write &write)
    {
        const auto id = nodeID;
        printVertex(write);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        write.start.visit(*this);
    }

    void DotVisualizer::operator()(Scatter &scatter)
    {
        const auto id = nodeID;
        printVertex(scatter);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        scatter.src.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        scatter.idxs.visit(*this);
    }

    void DotVisualizer::operator()(FunctionCall &call)
    {
        printVertex(call);
    }

    void DotVisualizer::operator()(Assign &assign)
    {
        const auto id = nodeID;
        printVertex(assign);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        assign.expr.visit(*this);
    }

    void DotVisualizer::operator()(Emit &emit)
    {
        const auto id = nodeID;
        printVertex(emit);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        emit.expr.visit(*this);
    }

    void DotVisualizer::operator()(Loop &loop)
    {
        const auto id = nodeID;
        printVertex(loop);

        for (const auto &stmt : loop.stms)
        {
            *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
            stmt.visit(*this);
        }

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        loop.pred.visit(*this);
    }

    void DotVisualizer::operator()(Add &add)
    {
        const auto id = nodeID;
        printVertex(add);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        add.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        add.rhs.visit(*this);
    }

    void DotVisualizer::operator()(Sub &sub)
    {
        const auto id = nodeID;
        printVertex(sub);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        sub.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        sub.rhs.visit(*this);
    }

    void DotVisualizer::operator()(Mul &mul)
    {
        const auto id = nodeID;
        printVertex(mul);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        mul.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        mul.rhs.visit(*this);
    }

    void DotVisualizer::operator()(Div &div)
    {
        const auto id = nodeID;
        printVertex(div);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        div.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        div.rhs.visit(*this);
    }

    void DotVisualizer::operator()(Mod &mod)
    {
        const auto id = nodeID;
        printVertex(mod);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        mod.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        mod.rhs.visit(*this);
    }

    void DotVisualizer::operator()(Eq &eq)
    {
        const auto id = nodeID;
        printVertex(eq);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        eq.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        eq.rhs.visit(*this);
    }

    void DotVisualizer::operator()(Neq &neq)
    {
        const auto id = nodeID;
        printVertex(neq);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        neq.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        neq.rhs.visit(*this);
    }

    void DotVisualizer::operator()(Le &le)
    {
        const auto id = nodeID;
        printVertex(le);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        le.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        le.rhs.visit(*this);
    }

    void DotVisualizer::operator()(Ge &ge)
    {
        const auto id = nodeID;
        printVertex(ge);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        ge.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        ge.rhs.visit(*this);
    }

    void DotVisualizer::operator()(Leq &leq)
    {
        const auto id = nodeID;
        printVertex(leq);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        leq.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        leq.rhs.visit(*this);
    }

    void DotVisualizer::operator()(Geq &geq)
    {
        const auto id = nodeID;
        printVertex(geq);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        geq.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        geq.rhs.visit(*this);
    }

    void DotVisualizer::operator()(And &anAnd)
    {
        const auto id = nodeID;
        printVertex(anAnd);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        anAnd.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        anAnd.rhs.visit(*this);
    }

    void DotVisualizer::operator()(Or &anOr)
    {
        const auto id = nodeID;
        printVertex(anOr);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        anOr.lhs.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        anOr.rhs.visit(*this);
    }

    void DotVisualizer::operator()(Not &aNot)
    {
        const auto id = nodeID;
        printVertex(aNot);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        aNot.param.visit(*this);
    }

    void DotVisualizer::operator()(IntConst &intConst)
    {
        printVertex(intConst);
    }

    void DotVisualizer::operator()(BooleanConst &boolConst)
    {
        printVertex(boolConst);
    }

    void DotVisualizer::operator()(FltConst &fltConst)
    {
        printVertex(fltConst);
    }

    void DotVisualizer::operator()(StrConst &strConst)
    {
        printVertex(strConst);
    }

    void DotVisualizer::operator()(Read &read)
    {
        const auto id = nodeID;
        printVertex(read);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        read.column.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        read.idx.visit(*this);
    }

    void DotVisualizer::operator()(Gather &gather)
    {
        const auto id = nodeID;
        printVertex(gather);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        gather.column.visit(*this);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        gather.idxs.visit(*this);
    }

    void DotVisualizer::operator()(Ref &param)
    {
        const auto id = nodeID;
        printVertex(param);
        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        param.ref.visit(*this);
    }

    void DotVisualizer::operator()(TupleGet &get)
    {
        printVertex(get);
    }

    void DotVisualizer::operator()(TupleCreate &create)
    {
        const auto id = nodeID;
        printVertex(create);

        for (const auto &elem : create.elems)
        {
            *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
            elem.visit(*this);
        }
    }


    void DotVisualizer::operator()(Selection &create) {
        const auto id = nodeID;
        printVertex(create);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        create.param.visit(*this);
    }

    void DotVisualizer::operator()(StatementWrapper &wrapper)
    {
        const auto id = nodeID;
        printVertex(wrapper);

        *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
        wrapper.expr.visit(*this);
    }

    void DotVisualizer::operator()(Fun &fun)
    {
        const auto id = nodeID;
        printVertex(fun);

        for (const auto &stmt : fun.body)
        {
            *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
            stmt.visit(*this);
        }
    }

    // FIXME: never called, instead Fun is called
    void DotVisualizer::operator()(Main &fun)
    {
        const auto id = nodeID;
        printVertex(fun);

        for (const auto &stmt : fun.body)
        {
            *os << fmt::format("n{} -> n{}\n", id, ++nodeID);
            stmt.visit(*this);
        }
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

    void DotVisualizer::printVertex(ASTNode &node)
    {
        *os << fmt::format("n{} [label=<<b>{}</b> <br/>", nodeID, node.type2string());
        node.print(*os);
        *os << ">]" << std::endl;
    }
} // namespace voila::ast

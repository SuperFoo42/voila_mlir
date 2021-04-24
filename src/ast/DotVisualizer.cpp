#include "ast/DotVisualizer.hpp"

void voila::ast::DotVisualizer::operator()(const voila::ast::AggrSum &sum)
{
    ASTVisitor::operator()(sum);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::AggrCnt &cnt)
{
    ASTVisitor::operator()(cnt);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::AggrMin &min)
{
    ASTVisitor::operator()(min);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::AggrMax &max)
{
    ASTVisitor::operator()(max);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::AggrAvg &avg)
{
    ASTVisitor::operator()(avg);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Write &write)
{
    ASTVisitor::operator()(write);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Scatter &scatter)
{
    ASTVisitor::operator()(scatter);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::FunctionCall &call)
{
    *os << fmt::format("n{} [label=<<b>{}</b> <br/>", nodeID++, call.type2string());
    call.print(*os);
    *os << ">]" << std::endl;
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Assign &assign)
{
    ASTVisitor::operator()(assign);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Emit &emit)
{
    ASTVisitor::operator()(emit);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Loop &loop)
{
    ASTVisitor::operator()(loop);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::StatementWrapper &wrapper)
{
    ASTVisitor::operator()(wrapper);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Add &add)
{
    ASTVisitor::operator()(add);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Sub &sub)
{
    ASTVisitor::operator()(sub);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Mul &mul)
{
    ASTVisitor::operator()(mul);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Div &div)
{
    ASTVisitor::operator()(div);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Mod &mod)
{
    ASTVisitor::operator()(mod);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Eq &eq)
{
    ASTVisitor::operator()(eq);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Neq &neq)
{
    ASTVisitor::operator()(neq);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Le &le)
{
    ASTVisitor::operator()(le);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Ge &ge)
{
    ASTVisitor::operator()(ge);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Leq &leq)
{
    ASTVisitor::operator()(leq);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Geq &geq)
{
    ASTVisitor::operator()(geq);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::And &anAnd)
{
    ASTVisitor::operator()(anAnd);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Or &anOr)
{
    ASTVisitor::operator()(anOr);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Not &aNot)
{
    ASTVisitor::operator()(aNot);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::IntConst &intConst)
{
    *os << fmt::format("n{} [label=<<b>{}</b> <br/>", nodeID++, intConst.type2string());
    intConst.print(*os);
    *os << ">]" << std::endl;
}
void voila::ast::DotVisualizer::operator()(const voila::ast::BooleanConst &boolConst)
{
    *os << fmt::format("n{} [label=<<b>{}</b> <br/>", nodeID++, boolConst.type2string());
    boolConst.print(*os);
    *os << ">]" << std::endl;
}
void voila::ast::DotVisualizer::operator()(const voila::ast::FltConst &fltConst)
{
    *os << fmt::format("n{} [label=<<b>{}</b> <br/>", nodeID++, fltConst.type2string());
    fltConst.print(*os);
    *os << ">]" << std::endl;
}
void voila::ast::DotVisualizer::operator()(const voila::ast::StrConst &strConst)
{
    *os << fmt::format("n{} [label=<<b>{}</b> <br/>", nodeID++, strConst.type2string());
    strConst.print(*os);
    *os << ">]" << std::endl;
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Read &read)
{
    ASTVisitor::operator()(read);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Gather &gather)
{
    ASTVisitor::operator()(gather);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::Ref &param)
{
    ASTVisitor::operator()(param);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::TupleGet &get)
{
    ASTVisitor::operator()(get);
}
void voila::ast::DotVisualizer::operator()(const voila::ast::TupleCreate &create)
{
    ASTVisitor::operator()(create);
}

void voila::ast::DotVisualizer::operator()(const voila::ast::Fun &fun)
{
    const auto id = nodeID;
    *os << fmt::format("n{} [label=<<b>{}</b> <br/>", nodeID++, fun.type2string());
    fun.print(*os);
    *os << ">]" << std::endl;

    for (const auto &stmt : fun.body)
    {
        *os << fmt::format("n{} -> n{}\n", id, nodeID);
        stmt.visit(*this);
    }
}

std::ostream &voila::ast::DotVisualizer::operator<<(std::ostream &out)
{
    out << fmt::format("digraph {} {\n", to_dot.name);
    os = &out;
    nodeID = 0;
    to_dot.visit(*this);
    os = nullptr;
    out << "}" << std::endl;
    return out;
}

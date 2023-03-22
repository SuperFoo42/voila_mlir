#include "DotVisualizer.hpp"
#include "ASTNodes.hpp"
#include "ast/ASTNodeVariant.hpp"
#include "ast/Add.hpp"                   // for Add
#include "ast/AggrAvg.hpp"               // for AggrAvg
#include "ast/AggrCnt.hpp"               // for AggrCnt
#include "ast/AggrMax.hpp"               // for AggrMax
#include "ast/AggrMin.hpp"               // for AggrMin
#include "ast/AggrSum.hpp"               // for AggrSum
#include "ast/And.hpp"                   // for And
#include "ast/Assign.hpp"                // for Assign
#include "ast/BooleanConst.hpp"          // for BooleanConst
#include "ast/Div.hpp"                   // for Div
#include "ast/Emit.hpp"                  // for Emit
#include "ast/Eq.hpp"                    // for Eq
#include "ast/FltConst.hpp"              // for FltConst
#include "ast/Fun.hpp"                   // for Fun
#include "ast/FunctionCall.hpp"          // for FunctionCall
#include "ast/Gather.hpp"                // for Gather
#include "ast/Ge.hpp"                    // for Ge
#include "ast/Geq.hpp"                   // for Geq
#include "ast/IntConst.hpp"              // for IntConst
#include "ast/Le.hpp"                    // for Le
#include "ast/Leq.hpp"                   // for Leq
#include "ast/Loop.hpp"                  // for Loop
#include "ast/Main.hpp"                  // for Main
#include "ast/Mod.hpp"                   // for Mod
#include "ast/Mul.hpp"                   // for Mul
#include "ast/Neq.hpp"                   // for Neq
#include "ast/Not.hpp"                   // for Not
#include "ast/Or.hpp"                    // for Or
#include "ast/Read.hpp"                  // for Read
#include "ast/Ref.hpp"                   // for Ref
#include "ast/Scatter.hpp"               // for Scatter
#include "ast/Selection.hpp"             // for Selection
#include "ast/StatementWrapper.hpp"      // for StatementWrapper
#include "ast/StrConst.hpp"              // for StrConst
#include "ast/Sub.hpp"                   // for Sub
#include "ast/Variable.hpp"              // for Variable
#include "ast/Write.hpp"                 // for Write
#include "llvm/Support/FormatVariadic.h" // for formatv, formatv_object
#include <variant>
#include <vector> // for vector

namespace voila::ast
{
    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<AggrSum> sum)
    {
        const auto id = nodeID;
        printVertex(sum);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, sum->src());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<AggrCnt> cnt)
    {
        const auto id = nodeID;
        printVertex(cnt);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, cnt->src());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<AggrMin> min)
    {
        const auto id = nodeID;
        printVertex(min);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, min->src());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<AggrMax> max)
    {
        const auto id = nodeID;
        printVertex(max);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, max->src());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<AggrAvg> avg)
    {
        const auto id = nodeID;
        printVertex(avg);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, avg->src());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Write> write)
    {
        const auto id = nodeID;
        printVertex(write);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, write->start());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Scatter> scatter)
    {
        const auto id = nodeID;
        printVertex(scatter);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, scatter->src());
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, scatter->idxs());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<FunctionCall> call) { printVertex(call); }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Assign> assign)
    {
        const auto id = nodeID;
        printVertex(assign);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        for (const auto &dest : assign->dests())
        {
            std::visit(*this, dest);
        }
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, assign->expr());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Emit> emit)
    {
        const auto id = nodeID;
        printVertex(emit);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        for (const auto &expr : emit->exprs())
            std::visit(*this, expr);
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Loop> loop)
    {
        const auto id = nodeID;
        printVertex<false>(loop);

        for (const auto &stmt : loop->stmts())
        {
            *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
            std::visit(*this, stmt);
        }

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, loop->pred());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Add> add)
    {
        const auto id = nodeID;
        printVertex(add);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, add->lhs());
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, add->rhs());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Sub> sub)
    {
        const auto id = nodeID;
        printVertex(sub);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, sub->lhs());
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, sub->rhs());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Mul> mul)
    {
        const auto id = nodeID;
        printVertex(mul);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, mul->lhs());
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, mul->rhs());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Div> div)
    {
        const auto id = nodeID;
        printVertex(div);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, div->lhs());
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, div->rhs());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Mod> mod)
    {
        const auto id = nodeID;
        printVertex(mod);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, mod->lhs());
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, mod->rhs());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Eq> eq)
    {
        const auto id = nodeID;
        printVertex(eq);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, eq->lhs());
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, eq->rhs());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Neq> neq)
    {
        const auto id = nodeID;
        printVertex(neq);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, neq->lhs());
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, neq->rhs());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Le> le)
    {
        const auto id = nodeID;
        printVertex(le);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, le->lhs());
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, le->rhs());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Ge> ge)
    {
        const auto id = nodeID;
        printVertex(ge);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, ge->lhs());
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, ge->rhs());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Leq> leq)
    {
        const auto id = nodeID;
        printVertex(leq);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, leq->lhs());
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, leq->rhs());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Geq> geq)
    {
        const auto id = nodeID;
        printVertex(geq);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, geq->lhs());
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, geq->rhs());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<And> anAnd)
    {
        const auto id = nodeID;
        printVertex(anAnd);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, anAnd->lhs());
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, anAnd->rhs());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Or> anOr)
    {
        const auto id = nodeID;
        printVertex(anOr);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, anOr->lhs());
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, anOr->rhs());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Not> aNot)
    {
        const auto id = nodeID;
        printVertex(aNot);
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, aNot->param());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<IntConst> intConst) { printVertex(intConst); }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<BooleanConst> boolConst) { printVertex(boolConst); }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<FltConst> fltConst) { printVertex(fltConst); }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<StrConst> strConst) { printVertex(strConst); }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Read> read)
    {
        const auto id = nodeID;
        printVertex(read);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, read->column());
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, read->idx());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Gather> gather)
    {
        const auto id = nodeID;
        printVertex(gather);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, gather->column());
        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, gather->idxs());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Ref> param) { printVertex(param); }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::monostate) { throw std::logic_error("Invalid node type monostate"); }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Selection> create)
    {
        const auto id = nodeID;
        printVertex(create);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, create->param());

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, create->pred());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<StatementWrapper> wrapper)
    {
        const auto id = nodeID;
        printVertex<false>(wrapper);

        *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
        std::visit(*this, wrapper->expr());
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Fun> fun)
    {
        const auto id = nodeID;
        printVertex(fun);

        for (auto &stmt : fun->body())
        {
            *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
            std::visit(*this, stmt);
        }
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Main> fun)
    {
        const auto id = nodeID;
        printVertex(fun);

        for (const auto &stmt : fun->body())
        {
            *os << llvm::formatv("n{0} -> n{1}\n", id, ++nodeID).str();
            std::visit(*this, stmt);
        }
    }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Variable> var) { printVertex(var); }

    std::ostream &operator<<(std::ostream &out, DotVisualizer &t)
    {
        out << "digraph " + t.to_dot->name() + "{\n";
        t.os = &out;
        t.nodeID = 0;
        t(t.to_dot);
        t.os = nullptr;
        out << "}" << std::endl;
        return out;
    }
    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Predicate>) { throw std::logic_error("Not implemented"); }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Insert>) { throw std::logic_error("Not implemented"); }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Lookup>) { throw std::logic_error("Not implemented"); }

    DotVisualizer::return_type DotVisualizer::visit_impl(std::shared_ptr<Hash>) { throw std::logic_error("Not implemented"); }

    template <bool infer_type> void DotVisualizer::printVertex(const ASTNodeVariant &node)
    {
        *os << llvm::formatv("n{0} [label=<<b>{1} <br/>", nodeID,
                             std::visit(overloaded{[](auto &n) -> std::string { return n->type2string(); },
                                                   [](std::monostate) { return std::string(); }},
                                        node))
                   .str();
        if (infer_type && inferer.has_value())
        {
            const auto type = inferer->get().get_type(node);
            // FIXME: overload does not work, need dynamic cast
            const auto res = std::dynamic_pointer_cast<FunctionType>(type);
            if (!res)
            {
                if (std::dynamic_pointer_cast<const ScalarType>(type))
                {
                    *os << *std::dynamic_pointer_cast<const ScalarType>(type);
                }
                else
                {
                    *os << *std::dynamic_pointer_cast<FunctionType>(type);
                }
            }
            else
            {
                *os << *res;
            }

            *os << "<br/>";
        }
        *os << "@"
            << std::visit(overloaded{[](auto &n) -> Location{ return n->get_location(); },
                                     [](std::monostate) -> Location { throw std::logic_error(""); }},
                          node)
            << "</b> <br/>";
        std::visit(overloaded{[&](auto &n) { n->print(*os); }, [](std::monostate) { throw std::logic_error(""); }},
                   node);
        *os << ">]" << std::endl;
    }
} // namespace voila::ast

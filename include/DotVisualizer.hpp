#pragma once
#include "ASTNodes.hpp"
#include "TypeInferer.hpp"
#include "Types.hpp"
#include "ast/ASTVisitor.hpp"
#include "llvm/Support/FormatVariadic.h"

#include <iostream>
#include <utility>

namespace voila::ast
{
    class DotVisualizer : public ASTVisitor
    {
      protected:
        Fun &to_dot;
        std::ostream *os;
        size_t nodeID;
        const std::optional<std::reference_wrapper<TypeInferer>> inferer;

        template<bool infer_type = false>
        void printVertex(const ASTNode &node)
        {
            *os << llvm::formatv("n{0} [label=<<b>{1} <br/>", nodeID, node.type2string()).str();
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
            *os << "@" << node.get_location() << "</b> <br/>";
            node.print(*os);
            *os << ">]" << std::endl;
        }

      public:
        explicit DotVisualizer(Fun &start,
                               const std::optional<std::reference_wrapper<TypeInferer>> &inferer = std::nullopt) :
            to_dot{start}, os{nullptr}, nodeID{0}, inferer{inferer}
        {
        }
        void operator()(const AggrSum &sum) final;
        void operator()(const AggrCnt &cnt) final;
        void operator()(const AggrMin &min) final;
        void operator()(const AggrMax &max) final;
        void operator()(const AggrAvg &avg) final;
        void operator()(const Write &write) final;
        void operator()(const Scatter &scatter) final;
        void operator()(const FunctionCall &call) final;
        void operator()(const Assign &assign) final;
        void operator()(const Emit &emit) final;
        void operator()(const Loop &loop) final;
        void operator()(const StatementWrapper &wrapper) final;
        void operator()(const Add &add) final;
        void operator()(const Sub &sub) final;
        void operator()(const Mul &mul) final;
        void operator()(const Div &div) final;
        void operator()(const Mod &mod) final;
        void operator()(const Eq &eq) final;
        void operator()(const Neq &neq) final;
        void operator()(const Le &le) final;
        void operator()(const Ge &ge) final;
        void operator()(const Leq &leq) final;
        void operator()(const Geq &geq) final;
        void operator()(const And &anAnd) final;
        void operator()(const Or &anOr) final;
        void operator()(const Not &aNot) final;
        void operator()(const IntConst &aConst) final;
        void operator()(const BooleanConst &aConst) final;
        void operator()(const FltConst &aConst) final;
        void operator()(const StrConst &aConst) final;
        void operator()(const Read &read) final;
        void operator()(const Gather &gather) final;
        void operator()(const Ref &param) final;
        void operator()(const TupleGet &get) final;
        void operator()(const TupleCreate &create) final;
        void operator()(const Main &create) final;
        void operator()(const Fun &create) final;
        void operator()(const Selection &create) final;
        void operator()(const Variable &var) final;

        friend std::ostream &operator<<(std::ostream &out, DotVisualizer &t);
    };
} // namespace voila::ast
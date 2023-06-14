#pragma once

#include "TypeInferer.hpp"               // for TypeInferer
#include "Types.hpp"                     // for operator<<, FunctionType (p...
#include "ast/ASTNode.hpp"               // for ASTNode
#include "ast/ASTVisitor.hpp"            // for ASTVisitor
#include "location.hpp"                  // for operator<<
#include "llvm/Support/FormatVariadic.h" // for formatv, formatv_object
#include <cstddef>                       // for size_t
#include <functional>                    // for reference_wrapper
#include <iostream>                      // for operator<<, ostream, basic_...
#include <memory>                        // for dynamic_pointer_cast, share...
#include <optional>                      // for optional, nullopt
#include <string>                        // for operator<<

namespace voila::ast
{

    class DotVisualizer : public ASTVisitor<DotVisualizer, void>
    {
      protected:
        std::shared_ptr<Fun> to_dot;
        std::ostream *os;
        size_t nodeID;
        const std::optional<std::reference_wrapper<TypeInferer>> inferer;

        template <bool infer_type = false> void printVertex(const ASTNodeVariant &node);

      public:
        explicit DotVisualizer(std::shared_ptr<Fun> start,
                               const std::optional<std::reference_wrapper<TypeInferer>> &inferer = std::nullopt)
            : to_dot{std::move(start)}, os{nullptr}, nodeID{0}, inferer{inferer}
        {
        }

        return_type visit_impl(std::shared_ptr<AggrSum> sum);

        return_type visit_impl(std::shared_ptr<AggrCnt> cnt);

        return_type visit_impl(std::shared_ptr<AggrMin> min);

        return_type visit_impl(std::shared_ptr<AggrMax> max);

        return_type visit_impl(std::shared_ptr<AggrAvg> avg);

        return_type visit_impl(std::shared_ptr<Write> write);

        return_type visit_impl(std::shared_ptr<Scatter> scatter);

        return_type visit_impl(std::shared_ptr<FunctionCall> call);

        return_type visit_impl(std::shared_ptr<Assign> assign);

        return_type visit_impl(std::shared_ptr<Emit> emit);

        return_type visit_impl(std::shared_ptr<Loop> loop);

        return_type visit_impl(std::shared_ptr<StatementWrapper> wrapper);

        return_type visit_impl(std::shared_ptr<Add> add);

        return_type visit_impl(std::shared_ptr<Sub> sub);

        return_type visit_impl(std::shared_ptr<Mul> mul);

        return_type visit_impl(std::shared_ptr<Div> div);

        return_type visit_impl(std::shared_ptr<Mod> mod);

        return_type visit_impl(std::shared_ptr<Eq> eq);

        return_type visit_impl(std::shared_ptr<Neq> neq);

        return_type visit_impl(std::shared_ptr<Le> le);

        return_type visit_impl(std::shared_ptr<Ge> ge);

        return_type visit_impl(std::shared_ptr<Leq> leq);

        return_type visit_impl(std::shared_ptr<Geq> geq);

        return_type visit_impl(std::shared_ptr<And> anAnd);

        return_type visit_impl(std::shared_ptr<Or> anOr);

        return_type visit_impl(std::shared_ptr<Predicate>);

        return_type visit_impl(std::shared_ptr<Insert>);

        return_type visit_impl(std::shared_ptr<Hash>);

        return_type visit_impl(std::shared_ptr<Lookup>);

        return_type visit_impl(std::shared_ptr<Not> aNot);

        return_type visit_impl(std::shared_ptr<IntConst> aConst);

        return_type visit_impl(std::shared_ptr<BooleanConst> aConst);

        return_type visit_impl(std::shared_ptr<FltConst> aConst);

        return_type visit_impl(std::shared_ptr<StrConst> aConst);

        return_type visit_impl(std::shared_ptr<Read> read);

        return_type visit_impl(std::shared_ptr<Gather> gather);

        return_type visit_impl(std::shared_ptr<Ref> param);

        return_type visit_impl(std::shared_ptr<Main> create);

        return_type visit_impl(std::shared_ptr<Fun> create);

        return_type visit_impl(std::shared_ptr<Selection> create);

        return_type visit_impl(std::shared_ptr<Variable> var);

        return_type visit_impl(std::shared_ptr<Load> var);

        return_type visit_impl(std::monostate);

        friend std::ostream &operator<<(std::ostream &out, DotVisualizer &t);
    };
} // namespace voila::ast
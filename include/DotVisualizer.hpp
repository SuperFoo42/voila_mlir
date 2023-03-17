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

    class DotVisualizer : public ASTVisitor<>
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

        void operator()(std::shared_ptr<AggrSum> sum) final;

        void operator()(std::shared_ptr<AggrCnt> cnt) final;

        void operator()(std::shared_ptr<AggrMin> min) final;

        void operator()(std::shared_ptr<AggrMax> max) final;

        void operator()(std::shared_ptr<AggrAvg> avg) final;

        void operator()(std::shared_ptr<Write> write) final;

        void operator()(std::shared_ptr<Scatter> scatter) final;

        void operator()(std::shared_ptr<FunctionCall> call) final;

        void operator()(std::shared_ptr<Assign> assign) final;

        void operator()(std::shared_ptr<Emit> emit) final;

        void operator()(std::shared_ptr<Loop> loop) final;

        void operator()(std::shared_ptr<StatementWrapper> wrapper) final;

        void operator()(std::shared_ptr<Add> add) final;

        void operator()(std::shared_ptr<Sub> sub) final;

        void operator()(std::shared_ptr<Mul> mul) final;

        void operator()(std::shared_ptr<Div> div) final;

        void operator()(std::shared_ptr<Mod> mod) final;

        void operator()(std::shared_ptr<Eq> eq) final;

        void operator()(std::shared_ptr<Neq> neq) final;

        void operator()(std::shared_ptr<Le> le) final;

        void operator()(std::shared_ptr<Ge> ge) final;

        void operator()(std::shared_ptr<Leq> leq) final;

        void operator()(std::shared_ptr<Geq> geq) final;

        void operator()(std::shared_ptr<And> anAnd) final;

        void operator()(std::shared_ptr<Or> anOr) final;

        void operator()(std::shared_ptr<Not> aNot) final;

        void operator()(std::shared_ptr<IntConst> aConst) final;

        void operator()(std::shared_ptr<BooleanConst> aConst) final;

        void operator()(std::shared_ptr<FltConst> aConst) final;

        void operator()(std::shared_ptr<StrConst> aConst) final;

        void operator()(std::shared_ptr<Read> read) final;

        void operator()(std::shared_ptr<Gather> gather) final;

        void operator()(std::shared_ptr<Ref> param) final;

        void operator()(std::shared_ptr<Main> create) final;

        void operator()(std::shared_ptr<Fun> create) final;

        void operator()(std::shared_ptr<Selection> create) final;

        void operator()(std::shared_ptr<Variable> var) final;
        using ASTVisitor<void>::operator();

        friend std::ostream &operator<<(std::ostream &out, DotVisualizer &t);
    };
} // namespace voila::ast
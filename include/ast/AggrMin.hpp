#pragma once
#include "Aggregation.hpp"
#include "Expression.hpp"
#include "ASTVisitor.hpp"
namespace voila::ast
{
    class AggrMin : public Aggregation
    {
      public:
        using Aggregation::Aggregation;
        using Aggregation::clone;

        [[nodiscard]] bool is_aggr_min() const final;

        AggrMin *as_aggr_min() final;

        [[nodiscard]] std::string type2string() const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;
    };
} // namespace voila::ast
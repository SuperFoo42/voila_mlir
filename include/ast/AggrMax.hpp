#pragma once
#include "Aggregation.hpp"
#include "Expression.hpp"
#include "ASTVisitor.hpp"
namespace voila::ast
{
    class AggrMax : public Aggregation
    {
      public:
        using Aggregation::Aggregation;

        [[nodiscard]] bool is_aggr_max() const final;

        AggrMax *as_aggr_max() final;

        [[nodiscard]] std::string type2string() const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;
    };
} // namespace voila::ast
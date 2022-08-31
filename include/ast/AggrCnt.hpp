#pragma once
#include "ASTVisitor.hpp"
#include "Aggregation.hpp"
#include "Expression.hpp"
namespace voila::ast
{
    class AggrCnt : public Aggregation
    {
      public:
        using Aggregation::Aggregation;
        using Aggregation::clone;

        [[nodiscard]] bool is_aggr_cnt() const final;

        AggrCnt *as_aggr_cnt() final;

        [[nodiscard]] std::string type2string() const final;
        void visit(ASTVisitor &visitor) const final;
        void visit(ASTVisitor &visitor) final;
    };
} // namespace voila::ast
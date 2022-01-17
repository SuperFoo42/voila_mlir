#pragma once

#include "ASTNode.hpp"
#include "PredicationUnsupportedException.hpp"

#include <optional>

namespace voila::ast
{
    class Selection;
    class Const;
    class Add;
    class Arithmetic;
    class Sub;
    class Mul;
    class Div;
    class Mod;
    class Comparison;
    class Eq;
    class Neq;
    class Le;
    class Ge;
    class Leq;
    class Geq;
    class And;
    class Or;
    class Not;
    class Logical;
    class IntConst;
    class BooleanConst;
    class FltConst;
    class StrConst;
    class Read;
    class Gather;
    class Ref;
    class TupleGet;
    class TupleCreate;
    class Expression;
    class Predicate;
    class Variable;
    class Aggregation;
    class AggrSum;
    class AggrCnt;
    class AggrMin;
    class AggrMax;
    class AggrAvg;
    class Hash;
    class Lookup;
    class Insert;
    class Scatter;
    class IExpression : public ASTNode
    {
      public:
        explicit IExpression(const Location loc) : ASTNode(loc) {}
        ~IExpression() override = default;

        virtual void set_predicate(Expression);

        virtual std::optional<Expression> get_predicate();

        // type checks
        [[nodiscard]] bool is_expr() const final;

        [[nodiscard]] virtual bool is_select() const;

        [[nodiscard]] virtual bool is_arithmetic() const;

        [[nodiscard]] virtual bool is_add() const;

        [[nodiscard]] virtual bool is_sub() const;

        [[nodiscard]] virtual bool is_scatter() const;

        [[nodiscard]] virtual bool is_mul() const;

        [[nodiscard]] virtual bool is_div() const;

        [[nodiscard]] virtual bool is_mod() const;

        [[nodiscard]] virtual bool is_comparison() const;

        [[nodiscard]] virtual bool is_geq() const;

        [[nodiscard]] virtual bool is_ge() const;

        [[nodiscard]] virtual bool is_leq() const;

        [[nodiscard]] virtual bool is_le() const;

        [[nodiscard]] virtual bool is_neq() const;

        [[nodiscard]] virtual bool is_eq() const;

        [[nodiscard]] virtual bool is_logical() const;

        [[nodiscard]] virtual bool is_unary() const;

        [[nodiscard]] virtual bool is_binary() const;

        [[nodiscard]] virtual bool is_and() const;

        [[nodiscard]] virtual bool is_or() const;

        [[nodiscard]] virtual bool is_not() const;

        [[nodiscard]] virtual bool is_string() const;

        [[nodiscard]] virtual bool is_float() const;

        [[nodiscard]] virtual bool is_integer() const;

        [[nodiscard]] virtual bool is_bool() const;

        [[nodiscard]] virtual bool is_const() const;

        [[nodiscard]] virtual bool is_read() const;

        [[nodiscard]] virtual bool is_gather() const;

        [[nodiscard]] virtual bool is_tuple_get() const;

        [[nodiscard]] virtual bool is_reference() const;

        [[nodiscard]] virtual bool is_tuple_create() const;

        [[nodiscard]] virtual bool is_predicate() const;

        [[nodiscard]] virtual bool is_variable() const;

        [[nodiscard]] virtual bool is_aggr() const;

        [[nodiscard]] virtual bool is_aggr_sum() const;

        [[nodiscard]] virtual bool is_aggr_cnt() const;

        [[nodiscard]] virtual bool is_aggr_min() const;

        [[nodiscard]] virtual bool is_aggr_max() const;

        [[nodiscard]] virtual bool is_aggr_avg() const;

        [[nodiscard]] virtual bool is_hash() const;

        [[nodiscard]] virtual bool is_lookup() const;

        [[nodiscard]] virtual bool is_insert() const;

        // casts
        virtual IExpression *as_expr();

        virtual Selection *as_select();

        virtual Arithmetic *as_arithmetic();

        virtual Add *as_add();

        virtual Sub *as_sub();

        virtual Mul *as_mul();

        virtual Div *as_div();

        virtual Mod *as_mod();

        virtual Comparison *as_comparison();

        virtual Geq *as_geq();

        virtual Ge *as_ge();

        virtual Leq *as_leq();

        virtual Le *as_le();

        virtual Neq *as_neq();

        virtual Eq *as_eq();

        virtual Logical *as_logical();

        virtual And *as_and();

        virtual Or *as_or();

        virtual Not *as_not();

        virtual StrConst *as_string();

        virtual FltConst *as_float();

        virtual IntConst *as_integer();

        virtual BooleanConst *as_bool();

        virtual Const *as_const();

        virtual Read *as_read();

        virtual Gather *as_gather();

        virtual TupleGet *as_tuple_get();

        [[nodiscard]] virtual const Ref *as_reference() const;

        virtual TupleCreate *as_tuple_create();

        virtual Predicate *as_predicate();

        virtual Variable *as_variable();

        virtual Aggregation *as_aggr();

        virtual AggrSum *as_aggr_sum();

        virtual AggrCnt *as_aggr_cnt();

        virtual AggrMin *as_aggr_min();

        virtual AggrMax *as_aggr_max();

        virtual AggrAvg *as_aggr_avg();

        virtual Hash *as_hash();

        virtual Lookup *as_lookup();

        virtual Insert *as_insert();

        virtual Scatter *as_scatter();
    };
} // namespace voila::ast

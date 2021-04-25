#pragma once

#include "ASTNode.hpp"
#include "PredicationUnsupportedException.hpp"

#include <fmt/core.h>

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

    class IExpression : public ASTNode
    {
      public:
        ~IExpression() override = default;

        virtual void predicate(Expression expr);

        // type checks
        [[nodiscard]] bool is_expr() const final;

        [[nodiscard]] virtual bool is_select() const;

        [[nodiscard]] virtual bool is_arithmetic() const;

        [[nodiscard]] virtual bool is_add() const;

        [[nodiscard]] virtual bool is_sub() const;

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

        virtual Ref *as_reference();

        virtual TupleCreate *as_tuple_create();
    };
} // namespace voila::ast

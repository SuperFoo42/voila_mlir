#pragma once

#include "ASTNode.hpp"

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

    class IExpression : ASTNode
    {
      public:
        virtual ~IExpression() = default;
        using ASTNode::print;
        using ASTNode::type2string;
        using ASTNode::visit;

        // type checks
        bool is_expr() const final
        {
            return true;
        }

        virtual bool is_select() const
        {
            return false;
        }

        virtual bool is_arithmetic() const
        {
            return false;
        }

        virtual bool is_add() const
        {
            return false;
        }

        virtual bool is_sub() const
        {
            return false;
        }

        virtual bool is_mul() const
        {
            return false;
        }

        virtual bool is_div() const
        {
            return false;
        }

        virtual bool is_mod() const
        {
            return false;
        }

        virtual bool is_comparison() const
        {
            return false;
        }

        virtual bool is_geq() const
        {
            return false;
        }

        virtual bool is_ge() const
        {
            return false;
        }

        virtual bool is_leq() const
        {
            return false;
        }

        virtual bool is_le() const
        {
            return false;
        }

        virtual bool is_neq() const
        {
            return false;
        }

        virtual bool is_eq() const
        {
            return false;
        }

        virtual bool is_logical() const
        {
            return false;
        }

        virtual bool is_unary() const
        {
            return false;
        }

        virtual bool is_binary() const
        {
            return false;
        }

        virtual bool is_and() const
        {
            return false;
        }

        virtual bool is_or() const
        {
            return false;
        }

        virtual bool is_not() const
        {
            return false;
        }

        virtual bool is_string() const
        {
            return false;
        }

        virtual bool is_float() const
        {
            return false;
        }

        virtual bool is_integer() const
        {
            return false;
        }

        virtual bool is_bool() const
        {
            return false;
        }

        virtual bool is_const() const
        {
            return false;
        }

        virtual bool is_read() const
        {
            return false;
        }

        virtual bool is_gather() const
        {
            return false;
        }

        virtual bool is_tuple_get() const
        {
            return false;
        }

        virtual bool is_reference() const
        {
            return false;
        }

        virtual bool is_tuple_create() const
        {
            return false;
        }

        // casts
        virtual IExpression *as_expr()
        {
            return this;
        }

        virtual Selection *as_select()
        {
            return nullptr;
        }

        virtual Arithmetic *as_arithmetic()
        {
            return nullptr;
        }

        virtual Add *as_add()
        {
            return nullptr;
        }

        virtual Sub *as_sub()
        {
            return nullptr;
        }

        virtual Mul *as_mul()
        {
            return nullptr;
        }

        virtual Div *as_div()
        {
            return nullptr;
        }

        virtual Mod *as_mod()
        {
            return nullptr;
        }

        virtual Comparison *as_comparison()
        {
            return nullptr;
        }

        virtual Geq *as_geq()
        {
            return nullptr;
        }

        virtual Ge *as_ge()
        {
            return nullptr;
        }

        virtual Leq *as_leq()
        {
            return nullptr;
        }

        virtual Le *as_le()
        {
            return nullptr;
        }

        virtual Neq *as_neq()
        {
            return nullptr;
        }

        virtual Eq *as_eq()
        {
            return nullptr;
        }

        virtual Logical *as_logical()
        {
            return nullptr;
        }

        virtual And *as_and()
        {
            return nullptr;
        }

        virtual Or *as_or()
        {
            return nullptr;
        }

        virtual Not *as_not()
        {
            return nullptr;
        }

        virtual StrConst *as_string()
        {
            return nullptr;
        }

        virtual FltConst *as_float()
        {
            return nullptr;
        }

        virtual IntConst *as_integer()
        {
            return nullptr;
        }

        virtual BooleanConst *as_bool()
        {
            return nullptr;
        }

        virtual Const *as_const()
        {
            return nullptr;
        }

        virtual Read *as_read()
        {
            return nullptr;
        }

        virtual Gather *as_gather()
        {
            return nullptr;
        }

        virtual TupleGet *as_tuple_get()
        {
            return nullptr;
        }

        virtual Ref *as_reference()
        {
            return nullptr;
        }


        virtual TupleCreate * as_tuple_create()
        {
            return nullptr;
        }
    };
} // namespace voila::ast

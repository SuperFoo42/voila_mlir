#pragma once
#include <iostream>
#include <string>

namespace voila::ast
{
    class ASTVisitor;
    class ASTNode
    {
      public:
        virtual void print(std::ostream &) const = 0;
        virtual std::string type2string() const = 0;
        virtual void visit(ASTVisitor &visitor);
        virtual ~ASTNode() {}

        virtual bool is_get_pos() const
        {
            return false;
        }

        virtual bool is_get_morsel() const
        {
            return false;
        }

        virtual bool is_aggr() const
        {
            return false;
        }

        virtual bool is_table_op() const
        {
            return false;
        }

        virtual bool is_constant() const
        {
            return false;
        }

        virtual bool is_cast() const
        {
            return false;
        }

        virtual bool is_select() const
        {
            return false;
        }

        virtual bool is_terminal() const
        {
            return false;
        }

        virtual bool is_tupleop() const
        {
            return false;
        }

        virtual bool is_expr() const
        {
            return false;
        }

        virtual bool is_stmt() const
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

        virtual bool is_loop() const
        {
            return false;
        }
    };
} // namespace voila::ast
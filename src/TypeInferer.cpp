#include "TypeInferer.hpp"

#include "IncompatibleTypesException.hpp"
#include "NonMatchingArityException.hpp"
#include "TypeNotInferedException.hpp"

#include <ast/Arithmetic.hpp>

namespace voila
{
    bool TypeInferer::convertible(DataType t1, DataType t2)
    {
        if (t1 == t2)
            return true;
        switch (t1)
        {
            case DataType::UNKNOWN:
                return true;
            case DataType::NUMERIC:
                return t2 == DataType::INT32 || t2 == DataType::INT64 || t2 == DataType::DBL || t2 == DataType::BOOL;
            case DataType::BOOL:
                return t2 == DataType::INT64 || t2 == DataType::INT32;
            case DataType::INT32:
                return t2 == DataType::INT64 || t2 == DataType::DBL;
            case DataType::INT64:
                return t2 == DataType::DBL;
            default:
                return false;
        }
    }

    Type &TypeInferer::get_type(const ast::ASTNode &node)
    {
        try
        {
            return types.at(&node);
        }
        catch (std::out_of_range &)
        {
            throw TypeNotInferedException();
        }
    }

    bool TypeInferer::compatible(DataType t1, DataType t2)
    {
        return convertible(t1, t2) || convertible(t2, t1);
    }

    static voila::DataType get_int_type(std::intmax_t i)
    {
        if (i < std::numeric_limits<uint_least32_t>::max())
        {
            return voila::DataType::INT32;
        }
        else
            return voila::DataType::INT64;
    }

    Type TypeInferer::unify([[maybe_unused]] Type t1, [[maybe_unused]] Type t2)
    {
        return Type(0);
    }

    DataType TypeInferer::convert(DataType t1, DataType t2)
    {
        if (convertible(t1, t2))
        {
            return t2;
        }
        else if (convertible(t2, t1))
        {
            return t1;
        }
        else
        {
            throw IncompatibleTypesException();
        }
    }

    void TypeInferer::operator()(const ast::Aggregation &aggregation)
    {
        ASTVisitor::operator()(aggregation);
    }

    void TypeInferer::operator()(const ast::Write &write)
    {
        ASTVisitor::operator()(write);
    }

    void TypeInferer::operator()(const ast::Scatter &scatter)
    {
        ASTVisitor::operator()(scatter);
    }

    void TypeInferer::operator()(const ast::FunctionCall &call)
    {
        ASTVisitor::operator()(call);
    }

    void TypeInferer::operator()(const ast::Variable &var)
    {
        types.emplace(&var, FunctionType(typeCnt++));
    }

    void TypeInferer::operator()(const ast::Assign &assign)
    {
        assert(assign.expr.as_expr());
        // TODO: type check variable
        const auto &exprType = types.at(assign.expr.as_expr());

        if (assign.dest.is_variable())
        {
            auto &varType = types.at(assign.dest.as_expr());
            varType.t = exprType.t; // TODO: isn't it more complex than that?
        }
        else
        {
            auto &varType = types.at(assign.dest.as_expr());

            if (!compatible(varType.t, exprType.t))
            {
                throw IncompatibleTypesException();
            }
        }

        types.emplace(&assign, FunctionType(typeCnt++, DataType::VOID));
    }

    void TypeInferer::operator()(const ast::Emit &emit)
    {
        ASTVisitor::operator()(emit);
    }

    void TypeInferer::operator()(const ast::Loop &loop)
    {
        ASTVisitor::operator()(loop);
    }

    void TypeInferer::operator()(const ast::Arithmetic &arithmetic)
    {
        auto &left_type = get_type(*arithmetic.lhs.as_expr());
        auto &right_type = get_type(*arithmetic.rhs.as_expr());
        if (left_type.ar != right_type.ar)
        {
            throw NonMatchingArityException();
        }
        if (!compatible(left_type.t, DataType::NUMERIC) || !compatible(right_type.t, DataType::NUMERIC))
        {
            throw IncompatibleTypesException();
        }

        if (left_type.t != right_type.t)
        {
            const auto newT = convert(left_type.t, right_type.t);
            left_type.t = newT;
            right_type.t = newT;
        }

        types.emplace(&arithmetic, Type(typeCnt++, left_type.t));
    }

    void TypeInferer::operator()(const ast::IntConst &aConst)
    {
        assert(!types.contains(&aConst));
        types.emplace(&aConst, Type(typeCnt++, get_int_type(aConst.val)));
    }

    void TypeInferer::operator()(const ast::BooleanConst &aConst)
    {
        assert(!types.contains(&aConst));
        types.emplace(&aConst, Type(typeCnt++, DataType::BOOL));
    }

    void TypeInferer::operator()(const ast::FltConst &aConst)
    {
        assert(!types.contains(&aConst));
        types.emplace(&aConst, Type(typeCnt++, DataType::DBL));
    }

    void TypeInferer::operator()(const ast::StrConst &aConst)
    {
        assert(!types.contains(&aConst));
        types.emplace(&aConst, Type(typeCnt++, DataType::STRING));
    }

    void TypeInferer::operator()(const ast::Read &read)
    {
        ASTVisitor::operator()(read);
    }

    void TypeInferer::operator()(const ast::Gather &gather)
    {
        ASTVisitor::operator()(gather);
    }

    void TypeInferer::operator()(const ast::Ref &param)
    {
        types.emplace(&param, Type(typeCnt++, types[param.ref.as_expr()].t, types[param.ref.as_expr()].ar));
    }

    void TypeInferer::operator()(const ast::TupleGet &get)
    {
        ASTVisitor::operator()(get);
    }

    void TypeInferer::operator()(const ast::TupleCreate &create)
    {
        ASTVisitor::operator()(create);
    }

    void TypeInferer::operator()(const ast::Fun &fun)
    {
        // TODO
        types.emplace(&fun, Type(typeCnt++));
    }

    void TypeInferer::operator()(const ast::Main &main)
    {
        // TODO
        types.emplace(&main, Type(typeCnt++));
    }

    void TypeInferer::operator()(const ast::Selection &selection)
    {
        ASTVisitor::operator()(selection);
    }

    void TypeInferer::operator()(const ast::Comparison &comparison)
    {
        auto &left_type = get_type(*comparison.lhs.as_expr());
        auto &right_type = get_type(*comparison.rhs.as_expr());
        if (left_type.ar != right_type.ar)
        {
            throw NonMatchingArityException();
        }
        if (!convertible(left_type.t, DataType::BOOL) || !convertible(right_type.t, DataType::BOOL))
        {
            throw IncompatibleTypesException();
        }

        if (left_type.t != right_type.t)
        {
            const auto newT = convert(left_type.t, right_type.t);
            left_type.t = newT;
            right_type.t = newT;
        }

        types.emplace(&comparison, Type(typeCnt++, DataType::BOOL));
    }

    void TypeInferer::operator()(const ast::Add &var)
    {
        TypeInferer::operator()(static_cast<const ast::Arithmetic &>(var));
    }
    void TypeInferer::operator()(const ast::Sub &sub)
    {
        TypeInferer::operator()(static_cast<const ast::Arithmetic &>(sub));
    }
    void TypeInferer::operator()(const ast::Mul &mul)
    {
        TypeInferer::operator()(static_cast<const ast::Arithmetic &>(mul));
    }
    void TypeInferer::operator()(const ast::Div &div1)
    {
        TypeInferer::operator()(static_cast<const ast::Arithmetic &>(div1));
    }
    void TypeInferer::operator()(const ast::Mod &mod)
    {
        TypeInferer::operator()(static_cast<const ast::Arithmetic &>(mod));
    }
    void TypeInferer::operator()(const ast::AggrSum &sum)
    {
        TypeInferer::operator()(static_cast<const ast::Aggregation &>(sum));
    }
    void TypeInferer::operator()(const ast::AggrCnt &cnt)
    {
        TypeInferer::operator()(static_cast<const ast::Aggregation &>(cnt));
    }
    void TypeInferer::operator()(const ast::AggrMin &aggrMin)
    {
        TypeInferer::operator()(static_cast<const ast::Aggregation &>(aggrMin));
    }
    void TypeInferer::operator()(const ast::AggrMax &aggrMax)
    {
        TypeInferer::operator()(static_cast<const ast::Aggregation &>(aggrMax));
    }
    void TypeInferer::operator()(const ast::AggrAvg &avg)
    {
        TypeInferer::operator()(static_cast<const ast::Aggregation &>(avg));
    }
    void TypeInferer::operator()(const ast::Eq &eq)
    {
        TypeInferer::operator()(static_cast<const ast::Comparison &>(eq));
    }
    void TypeInferer::operator()(const ast::Neq &neq)
    {
        TypeInferer::operator()(static_cast<const ast::Comparison &>(neq));
    }
    void TypeInferer::operator()(const ast::Le &le)
    {
        TypeInferer::operator()(static_cast<const ast::Comparison &>(le));
    }
    void TypeInferer::operator()(const ast::Ge &ge)
    {
        TypeInferer::operator()(static_cast<const ast::Comparison &>(ge));
    }
    void TypeInferer::operator()(const ast::Leq &leq)
    {
        TypeInferer::operator()(static_cast<const ast::Comparison &>(leq));
    }
    void TypeInferer::operator()(const ast::Geq &geq)
    {
        TypeInferer::operator()(static_cast<const ast::Comparison &>(geq));
    }
    void TypeInferer::operator()(const ast::And &anAnd)
    {
        auto &left_type = get_type(*anAnd.lhs.as_expr());
        auto &right_type = get_type(*anAnd.rhs.as_expr());
        if (left_type.ar != right_type.ar)
        {
            throw NonMatchingArityException();
        }
        if (!convertible(left_type.t, DataType::BOOL) || !convertible(right_type.t, DataType::BOOL))
        {
            throw IncompatibleTypesException();
        }

        if (left_type.t != right_type.t)
        {
            const auto newT = convert(left_type.t, right_type.t);
            left_type.t = newT;
            right_type.t = newT;
        }

        types.emplace(&anAnd, Type(typeCnt++, DataType::BOOL));
    }
    void TypeInferer::operator()(const ast::Or &anOr)
    {
        auto &left_type = get_type(*anOr.lhs.as_expr());
        auto &right_type = get_type(*anOr.rhs.as_expr());
        if (left_type.ar != right_type.ar)
        {
            throw NonMatchingArityException();
        }
        if (!convertible(left_type.t, DataType::BOOL) || !convertible(right_type.t, DataType::BOOL))
        {
            throw IncompatibleTypesException();
        }

        if (left_type.t != right_type.t)
        {
            const auto newT = convert(left_type.t, right_type.t);
            left_type.t = newT;
            right_type.t = newT;
        }

        types.emplace(&anOr, Type(typeCnt++, DataType::BOOL));
    }
    void TypeInferer::operator()(const ast::Not &aNot)
    {
        auto &type = get_type(*aNot.param.as_expr());

        if (!convertible(type.t, DataType::BOOL))
        {
            throw IncompatibleTypesException();
        }

        types.emplace(&aNot, Type(typeCnt++, DataType::BOOL));
    }
} // namespace voila
#include "TypeInferer.hpp"

#include "IncompatibleTypesException.hpp"
#include "NonMatchingArityException.hpp"
#include "TypeNotInferedException.hpp"
#include "ast/Arithmetic.hpp"

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

    /**
     * Add simple type
     * @param node
     * @param t
     * @param ar
     */
    void
    TypeInferer::insertNewType(const ast::ASTNode &node, const DataType t = DataType::UNKNOWN, const Arity ar = Arity())
    {
        typeIDs.try_emplace(&node, types.size());
        types.push_back(std::make_unique<Type>(types.size(), t, ar));
    }

    /**
     * Add function type
     * @param node
     * @param returnT
     * @param returnAr
     * @param typeParamIDs
     */
    void TypeInferer::insertNewFuncType(const ast::ASTNode &node,
                                        std::vector<size_t> typeParamIDs = {},
                                        const DataType returnT = DataType::UNKNOWN,
                                        const Arity returnAr = Arity())
    {
        typeIDs.try_emplace(&node, types.size());
        types.push_back(std::make_unique<FunctionType>(types.size(), std::move(typeParamIDs), returnT, returnAr));
    }

    size_t TypeInferer::get_type_id(const ast::Expression &node)
    {
        try
        {
            return typeIDs.at(node.as_expr());
        }
        catch (std::out_of_range &)
        {
            throw TypeNotInferedException();
        }
    }

    size_t TypeInferer::get_type_id(const ast::Statement &node)
    {
        try
        {
            return typeIDs.at(node.as_stmt());
        }
        catch (std::out_of_range &)
        {
            throw TypeNotInferedException();
        }
    }

    size_t TypeInferer::get_type_id(const ast::ASTNode &node)
    {
        try
        {
            return typeIDs.at(&node);
        }
        catch (std::out_of_range &)
        {
            throw TypeNotInferedException();
        }
    }

    Type &TypeInferer::get_type(const ast::Expression &node) const
    {
        try
        {
            return *types.at(typeIDs.at(node.as_expr()));
        }
        catch (std::out_of_range &)
        {
            throw TypeNotInferedException();
        }
    }

    Type &TypeInferer::get_type(const ast::Statement &node) const
    {
        try
        {
            return *types.at(typeIDs.at(node.as_stmt()));
        }
        catch (std::out_of_range &)
        {
            throw TypeNotInferedException();
        }
    }

    Type &TypeInferer::get_type(const ast::ASTNode &node) const
    {
        try
        {
            return *types.at(typeIDs.at(&node));
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

    void TypeInferer::unify(const ast::ASTNode &t1, const ast::ASTNode &t2)
    {
        // TODO
        typeIDs.emplace(&t1, get_type_id(t2));
    }

    void TypeInferer::unify(const ast::ASTNode &t1, const ast::Expression &t2)
    {
        // TODO
        typeIDs.emplace(&t1, get_type_id(t2));
    }

    void TypeInferer::unify(const ast::ASTNode &t1, const ast::Statement &t2)
    {
        // TODO
        if (!typeIDs.contains(&t1))
        {
            typeIDs.emplace(&t1, get_type_id(t2));
        }
        else
        {
            // TODO: type checking
            typeIDs[&t1] = get_type_id(t2);
        }
    }

    void TypeInferer::unify(const ast::Expression &t1, const ast::Expression &t2)
    {
        if (!typeIDs.contains(t1.as_expr()))
        {
            typeIDs.emplace(t1.as_expr(), get_type_id(t2));
        }
        else
        {
            // TODO: type checking
            typeIDs[t1.as_expr()] = get_type_id(t2);
        }
    }

    void TypeInferer::unify(const ast::Statement &t1, const ast::Statement &t2)
    {
        if (!typeIDs.contains(t1.as_stmt()))
        {
            typeIDs.emplace(t1.as_stmt(), get_type_id(t2));
        }
        else
        {
            // TODO: type checking
            typeIDs[t1.as_stmt()] = get_type_id(t2);
        }
    }

    void TypeInferer::operator()(const ast::Aggregation &aggregation)
    {
        //TODO
        ASTVisitor::operator()(aggregation);
    }

    void TypeInferer::operator()(const ast::Write &write)
    {
        if (!compatible(get_type(write.src).t, get_type(write.dest).t))
            throw IncompatibleTypesException();
        if (!compatible(get_type(write.start).t, DataType::INT64))
            throw IncompatibleTypesException();

        // TODO: unification

        insertNewFuncType(write, {get_type_id(write.src), get_type_id(write.dest), get_type_id(write.start)},
                          DataType::VOID);
    }

    void TypeInferer::operator()(const ast::Scatter &scatter)
    {
        if (!compatible(get_type(scatter.src).t, get_type(scatter.dest).t))
            throw IncompatibleTypesException();
        if (!compatible(get_type(scatter.idxs).t, DataType::INT64))
            throw IncompatibleTypesException();

        // TODO: unification

        insertNewFuncType(scatter, {get_type_id(scatter.src), get_type_id(scatter.dest), get_type_id(scatter.idxs)},
                          DataType::VOID);
    }

    void TypeInferer::operator()(const ast::FunctionCall &call)
    {
        std::vector<size_t> argIds;
        std::transform(
            call.args.begin(), call.args.end(),
            std::back_inserter(argIds), [&](const auto &elem) -> auto { return get_type_id(elem); });
        insertNewFuncType(call, argIds); // TODO
    }

    void TypeInferer::operator()(const ast::Variable &var)
    {
        insertNewType(var);
    }

    void TypeInferer::operator()(const ast::Assign &assign)
    {
        assert(assign.expr.as_expr());

        if (assign.dest.is_reference() && !compatible(get_type(assign.dest).t, get_type(assign.expr).t))
        {
            throw IncompatibleTypesException();
        }

        unify(assign.dest, assign.expr);

        insertNewFuncType(assign, {get_type_id(assign.dest), get_type_id(assign.expr)}, DataType::VOID);
    }

    void TypeInferer::operator()(const ast::Emit &emit)
    {
        //TODO
        ASTVisitor::operator()(emit);
    }

    void TypeInferer::operator()(const ast::Loop &loop)
    {
        if (!compatible(get_type(loop.pred).t, DataType::BOOL))
        {
            throw IncompatibleTypesException();
        }
    }

    void TypeInferer::operator()(const ast::Arithmetic &arithmetic)
    {
        const auto &left_type = get_type(arithmetic.lhs);
        const auto &right_type = get_type(arithmetic.rhs);
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
            unify(arithmetic.lhs, arithmetic.rhs);
        }

        insertNewFuncType(arithmetic, {get_type_id(arithmetic.lhs), get_type_id(arithmetic.rhs)},
                          get_type(arithmetic.lhs).t, left_type.ar);
    }

    void TypeInferer::operator()(const ast::IntConst &aConst)
    {
        assert(!typeIDs.contains(&aConst));
        insertNewType(aConst, get_int_type(aConst.val));
    }

    void TypeInferer::operator()(const ast::BooleanConst &aConst)
    {
        assert(!typeIDs.contains(&aConst));
        insertNewType(aConst, DataType::BOOL);
    }

    void TypeInferer::operator()(const ast::FltConst &aConst)
    {
        assert(!typeIDs.contains(&aConst));
        insertNewType(aConst, DataType::DBL);
    }

    void TypeInferer::operator()(const ast::StrConst &aConst)
    {
        assert(!typeIDs.contains(&aConst));
        insertNewType(aConst, DataType::STRING);
    }

    void TypeInferer::operator()(const ast::Read &read)
    {
        if (!compatible(get_type(read.idx).t, DataType::INT64))
            throw IncompatibleTypesException();
        if (compatible(get_type(read.column).t, DataType::VOID))
            throw IncompatibleTypesException();

        insertNewFuncType(read, {get_type_id(read.column), get_type_id(read.idx)}, DataType::VOID);
    }

    void TypeInferer::operator()(const ast::Gather &gather)
    {
        if (!compatible(get_type(gather.idxs).t, DataType::INT64))
            throw IncompatibleTypesException();
        if (compatible(get_type(gather.column).t, DataType::VOID))
            throw IncompatibleTypesException();

        insertNewFuncType(gather, {get_type_id(gather.column), get_type_id(gather.idxs)}, DataType::VOID);
    }

    void TypeInferer::operator()(const ast::Ref &param)
    {
        unify(param, param.ref);
    }

    void TypeInferer::operator()(const ast::TupleGet &)
    {
        // TODO: check expr list
        // insertNewType(get, get_type(get.expr));
    }

    void TypeInferer::operator()(const ast::TupleCreate &create)
    {
        // TODO
        ASTVisitor::operator()(create);
    }

    void TypeInferer::operator()(const ast::Fun &fun)
    {
        std::vector<size_t> argIds;
        std::transform(
            fun.args.begin(), fun.args.end(),
            std::back_inserter(argIds), [&](const auto &elem) -> auto { return get_type_id(elem); });
        insertNewFuncType(fun, argIds);
    }

    void TypeInferer::operator()(const ast::Main &main)
    {
        std::vector<size_t> argIds;
        std::transform(
            main.args.begin(), main.args.end(),
            std::back_inserter(argIds), [&](const auto &elem) -> auto { return get_type_id(elem); });
        insertNewFuncType(main, argIds);
    }

    void TypeInferer::operator()(const ast::Selection &selection)
    {
        auto &type = get_type(*selection.param.as_expr());

        if (!convertible(get_type(selection.pred).t, DataType::BOOL))
        {
            throw IncompatibleTypesException();
        }
        else
        {
            get_type(selection.pred).t = DataType::BOOL;
        }

        insertNewFuncType(selection, {get_type_id(selection.param), get_type_id(selection.param)}, type.t);
    }

    void TypeInferer::operator()(const ast::Comparison &comparison)
    {
        const auto &left_type = get_type(*comparison.lhs.as_expr());
        const auto &right_type = get_type(*comparison.rhs.as_expr());
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
            unify(comparison.lhs, comparison.rhs);
        }

        insertNewType(comparison, DataType::BOOL);
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
        const auto &left_type = get_type(*anAnd.lhs.as_expr());
        const auto &right_type = get_type(*anAnd.rhs.as_expr());
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
            unify(anAnd.lhs, anAnd.rhs);
        }

        insertNewType(anAnd, DataType::BOOL);
    }
    void TypeInferer::operator()(const ast::Or &anOr)
    {
        const auto &left_type = get_type(*anOr.lhs.as_expr());
        const auto &right_type = get_type(*anOr.rhs.as_expr());
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
            unify(anOr.lhs, anOr.rhs);
        }

        insertNewType(anOr, DataType::BOOL);
    }
    void TypeInferer::operator()(const ast::Not &aNot)
    {
        auto &type = get_type(*aNot.param.as_expr());

        if (!convertible(type.t, DataType::BOOL))
        {
            throw IncompatibleTypesException();
        }
        else
        {
            type.t = DataType::BOOL;
        }

        insertNewType(aNot, DataType::BOOL);
    }
} // namespace voila
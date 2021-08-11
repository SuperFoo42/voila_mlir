#include "TypeInferer.hpp"

#include "IncompatibleTypesException.hpp"
#include "NonMatchingArityException.hpp"
#include "NotInferedException.hpp"
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

    void TypeInferer::insertNewFuncType(const ast::ASTNode &node,
                                        std::vector<size_t> typeParamIDs,
                                        const size_t returnTypeID)
    {
        typeIDs.try_emplace(&node, types.size());
        types.push_back(std::make_unique<FunctionType>(types.size(), std::move(typeParamIDs), types.at(returnTypeID)->t,
                                                       types.at(returnTypeID)->ar));
    }

    size_t TypeInferer::get_type_id(const ast::Expression &node)
    {
        try
        {
            if (node.is_reference())
                return typeIDs.at(node.as_reference()->ref.as_expr());
            else
                return typeIDs.at(node.as_expr());
        }
        catch (std::out_of_range &)
        {
            throw NotInferedException();
        }
    }

    size_t TypeInferer::get_type_id(const ast::Statement &node)
    {
        try
        {
            if (node.is_statement_wrapper())
                return get_type_id(node.as_statement_wrapper()->expr);
            else
                return typeIDs.at(node.as_stmt());
        }
        catch (std::out_of_range &)
        {
            throw NotInferedException();
        }
    }

    size_t TypeInferer::get_type_id(const ast::ASTNode &node)
    {
        try
        {
            if (dynamic_cast<const ast::Ref *>(&node))
                return typeIDs.at(dynamic_cast<const ast::Ref *>(&node)->ref.as_expr());
            else
                return typeIDs.at(&node);
        }
        catch (std::out_of_range &)
        {
            throw NotInferedException();
        }
    }

    Type &TypeInferer::get_type(const ast::Expression &node) const
    {
        ast::IExpression *tmp = node.as_expr();
        if (node.is_reference()) // resolve reference
            tmp = node.as_reference()->ref.as_expr();
        try
        {
            return *types.at(typeIDs.at(tmp));
        }
        catch (std::out_of_range &)
        {
            throw NotInferedException();
        }
    }

    Type &TypeInferer::get_type(const ast::Statement &node) const
    {
        try
        {
            if (node.is_statement_wrapper())
                return *types.at(typeIDs.at(node.as_statement_wrapper()->expr.as_expr()));
            else
                return *types.at(typeIDs.at(node.as_stmt()));
        }
        catch (std::out_of_range &)
        {
            throw NotInferedException();
        }
    }

    Type &TypeInferer::get_type(const ast::ASTNode &node) const
    {
        try
        {
            if (dynamic_cast<const ast::Ref *>(&node))
                return *types.at(typeIDs.at(dynamic_cast<const ast::Ref *>(&node)->ref.as_expr()));
            else
                return *types.at(typeIDs.at(&node));
        }
        catch (std::out_of_range &)
        {
            throw NotInferedException();
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
        {
            return voila::DataType::INT64;
        }
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
        const ast::ASTNode *tmp1 = &t1, *tmp2 = &t2;
        if (dynamic_cast<const ast::Ref *>(&t1))
        {
            tmp1 = dynamic_cast<const ast::Ref *>(&t1)->ref.as_expr();
        }
        if (dynamic_cast<const ast::Ref *>(&t2))
        {
            tmp2 = dynamic_cast<const ast::Ref *>(&t2)->ref.as_expr();
        }
        unify(tmp1, tmp2);
    }

    void TypeInferer::unify(const ast::ASTNode &t1, const ast::Expression &t2)
    {
        if (t2.is_reference())
            unify(&t1, t2.as_reference()->ref.as_expr());
        else
            unify(&t1, t2.as_expr());
    }

    void TypeInferer::unify(const ast::ASTNode *t1, const ast::ASTNode *t2)
    {
        // TODO
        if (typeIDs.contains(t1) && !typeIDs.contains(t2))
        {
            typeIDs.emplace(t2, get_type_id(*t1));
        }
        else if (!typeIDs.contains(t1) && typeIDs.contains(t2))
        {
            typeIDs.emplace(t1, get_type_id(*t2));
        }
        else if (typeIDs.contains(t1) && typeIDs.contains(t2))
        {
            if (convertible(types[typeIDs[t1]]->t, types[typeIDs[t2]]->t))
            {
                typeIDs[t1] = typeIDs[t2];
            }
            else if (convertible(types[typeIDs[t2]]->t, types[typeIDs[t1]]->t))
            {
                typeIDs[t2] = typeIDs[t1];
            }
            else
            {
                throw IncompatibleTypesException();
            }
        }
        else
        {
            throw NotInferedException();
        }
    }

    void TypeInferer::unify(const ast::Expression &t1, const ast::Statement &t2)
    {
        ast::ASTNode *tmp;
        if (t2.is_statement_wrapper())
        {
            tmp = t2.as_statement_wrapper()->expr.as_expr();
        }
        else
        {
            tmp = t2.as_stmt();
        }
        if (t1.is_reference())
            typeIDs.insert_or_assign(t1.as_reference()->ref.as_expr(), get_type_id(*tmp));
        else
            typeIDs.insert_or_assign(t1.as_expr(), get_type_id(*tmp));
    }

    void TypeInferer::unify(const ast::ASTNode &t1, const ast::Statement &t2)
    {
        if (dynamic_cast<const ast::Ref *>(&t1))
            typeIDs.insert_or_assign(dynamic_cast<const ast::Ref *>(&t1)->ref.as_expr(), get_type_id(t2));
        else
            typeIDs.insert_or_assign(&t1, get_type_id(t2));
    }

    void TypeInferer::unify(const ast::Expression &t1, const ast::Expression &t2)
    {
        auto *tmp1 = t1.as_expr();
        auto *tmp2 = t2.as_expr();
        if (t1.is_reference())
            tmp1 = t1.as_reference()->ref.as_expr();
        if (t2.is_reference())
            tmp2 = t2.as_reference()->ref.as_expr();
        unify(tmp1, tmp2);
    }

    void TypeInferer::unify(const ast::Statement &t1, const ast::Statement &t2)
    {
        unify(t1.as_stmt(), t2.as_stmt());
    }

    void TypeInferer::operator()(const ast::Aggregation &aggregation)
    {
        // TODO
        aggregation.src.visit(*this);
        insertNewFuncType(aggregation, {get_type_id(aggregation.src)}, DataType::INT64, Arity(1));
    }

    void TypeInferer::operator()(const ast::Write &write)
    {
        write.start.visit(*this);
        write.src.visit(*this);
        write.dest.visit(*this);

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
        scatter.src.visit(*this);
        scatter.idxs.visit(*this);
        scatter.dest.visit(*this);

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
        for (auto &arg : call.args)
        {
            arg.visit(*this);
        }

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
        assign.dest.visit(*this);
        assign.expr.visit(*this);

        assert(assign.expr.is_function_call() || assign.expr.is_statement_wrapper());

        if (assign.dest.is_reference() &&
            !compatible(get_type(assign.dest.as_reference()->ref).t, get_type(assign.expr).t))
        {
            throw IncompatibleTypesException();
        }

        if (assign.expr.is_statement_wrapper())
            unify(assign.dest, assign.expr.as_statement_wrapper()->expr);
        else
            unify(assign.dest, assign.expr);

        insertNewFuncType(assign, {get_type_id(assign.dest), get_type_id(assign.expr)}, DataType::VOID);
    }

    void TypeInferer::operator()(const ast::Emit &emit)
    {
        emit.expr.visit(*this);

        typeIDs.try_emplace(&emit, get_type_id(emit.expr));
    }

    void TypeInferer::operator()(const ast::Loop &loop)
    {
        loop.pred.visit(*this);

        for (auto &stmt : loop.stms)
        {
            stmt.visit(*this);
        }

        if (!compatible(get_type(loop.pred).t, DataType::BOOL))
        {
            throw IncompatibleTypesException();
        }
    }

    void TypeInferer::operator()(const ast::Hash &hash)
    {
        //TODO: check same arities
        for (auto &elem : hash.items)
            elem.visit(*this);

        std::vector<size_t> type_ids;
        for (const auto &elem : hash.items)
            type_ids.push_back(get_type_id(elem));

        insertNewFuncType(hash, type_ids, DataType::INT64, get_type(hash.items.front()).ar);
    }

    void TypeInferer::operator()(const ast::Arithmetic &arithmetic)
    {
        arithmetic.lhs.visit(*this);
        arithmetic.rhs.visit(*this);

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
        read.column.visit(*this);
        read.idx.visit(*this);

        if (!compatible(get_type(read.idx).t, DataType::INT64))
            throw IncompatibleTypesException();
        if (get_type(read.column).t == DataType::VOID)
            throw IncompatibleTypesException();

        insertNewFuncType(read, {get_type_id(read.column), get_type_id(read.idx)}, DataType::VOID);
    }

    void TypeInferer::operator()(const ast::Gather &gather)
    {
        gather.column.visit(*this);
        gather.idxs.visit(*this);

        if (!compatible(get_type(gather.idxs).t, DataType::INT64))
            throw IncompatibleTypesException();
        if (get_type(gather.column).t == DataType::VOID)
            throw IncompatibleTypesException();

        insertNewFuncType(gather, {get_type_id(gather.column), get_type_id(gather.idxs)}, DataType::VOID);
    }

    void TypeInferer::operator()(const ast::Ref &)
    {
        // do not infer type, just act as a wrapper around variable
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
        // TODO: clear infered types at start of new function?
        for (auto &arg : fun.args) // infer function args
        {
            arg.visit(*this);
        }
        for (auto &stmt : fun.body) // infer body types
        {
            stmt.visit(*this);
        }
        std::vector<size_t> argIds;
        std::transform(
            fun.args.begin(), fun.args.end(),
            std::back_inserter(argIds), [&](const auto &elem) -> auto { return get_type_id(elem); });
        if (fun.result.has_value())
            insertNewFuncType(fun, argIds, get_type_id(*fun.result));
        else
            insertNewFuncType(fun, argIds, DataType::VOID);
    }

    void TypeInferer::operator()(const ast::Main &main)
    {
        // TODO: clear infered types at start of new function?
        // do not infer function args, they have to be specified by user before further inference

        for (auto &stmt : main.body) // infer body types
        {
            stmt.visit(*this);
        }

        std::vector<size_t> argIds;
        std::transform(
            main.args.begin(), main.args.end(),
            std::back_inserter(argIds), [&](const auto &elem) -> auto { return get_type_id(elem); });
        if (main.result.has_value())
        {
            insertNewFuncType(main, argIds, get_type_id(*main.result));
        }
        else
        {
            insertNewFuncType(main, argIds, DataType::VOID);
        }
    }

    void TypeInferer::operator()(const ast::Selection &selection)
    {
        selection.pred.visit(*this);
        selection.param.visit(*this);

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

    void TypeInferer::operator()(const ast::Lookup &lookup)
    {
        lookup.keys.visit(*this);
        lookup.table.visit(*this);

        // actually, return type is IndexType
        insertNewFuncType(lookup, {get_type_id(lookup.table), get_type_id(lookup.keys)}, DataType::INT64,
                          get_type(lookup.keys).ar);
    }
    void TypeInferer::operator()(const ast::Insert &insert)
    {
        insert.keys.visit(*this);
        insert.values.visit(*this);

        // actually, return type is IndexType
        insertNewFuncType(insert, {get_type_id(insert.keys),get_type_id(insert.values)}, get_type(insert.values).t, get_type(insert.keys).ar);
    }

    void TypeInferer::operator()(const ast::Predicate &pred)
    {
        pred.expr.visit(*this);

        auto &type = get_type(*pred.expr.as_expr());

        if (!convertible(get_type(pred.expr).t, DataType::BOOL))
        {
            throw IncompatibleTypesException();
        }
        else
        {
            get_type(pred.expr).t = DataType::BOOL;
        }

        insertNewFuncType(pred, {get_type_id(pred.expr), get_type_id(pred.expr)}, type.t);
    }

    void TypeInferer::operator()(const ast::Comparison &comparison)
    {
        comparison.lhs.visit(*this);
        comparison.rhs.visit(*this);

        auto &left_type = get_type(*comparison.lhs.as_expr());
        auto &right_type = get_type(*comparison.rhs.as_expr());
        // TODO: insert for all binary preds
        if (left_type.ar.is_undef() xor right_type.ar.is_undef())
        {
            if (left_type.ar.is_undef())
            {
                left_type.ar = right_type.ar;
            }
            else
            {
                right_type.ar = left_type.ar;
            }
        }
        if (left_type.ar != right_type.ar)
        {
            throw NonMatchingArityException();
        }
        /*        if (!convertible(left_type.t, DataType::BOOL) || !convertible(right_type.t, DataType::BOOL))
                {
                    throw IncompatibleTypesException();
                }*/

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
        anAnd.lhs.visit(*this);
        anAnd.rhs.visit(*this);

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
        anOr.lhs.visit(*this);
        anOr.rhs.visit(*this);

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
        aNot.param.visit(*this);

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

    void TypeInferer::operator()(const ast::StatementWrapper &wrapper)
    {
        wrapper.expr.visit(*this);
    }

    void TypeInferer::set_arity(const ast::ASTNode *const node, const size_t ar)
    {
        types.at(typeIDs.at(node))->ar = Arity(ar);
    }

    void TypeInferer::set_type(const ast::ASTNode *const node, const DataType type)
    {
        types.at(typeIDs.at(node))->t = type;
    }

} // namespace voila
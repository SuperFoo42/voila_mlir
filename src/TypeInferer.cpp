#include "TypeInferer.hpp"

#include "IncompatibleTypesException.hpp"
#include "NonMatchingArityException.hpp"
#include "NotInferedException.hpp"
#include "ast/Arithmetic.hpp"

#include <utility>
#include <range/v3/all.hpp>

namespace voila {
    bool TypeInferer::convertible(DataType t1, DataType t2) {
        if (t1 == t2)
            return true;
        switch (t1) {
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
    TypeInferer::insertNewType(const ast::ASTNode &node, const DataType t = DataType::UNKNOWN,
                               const Arity ar = Arity()) {
        typeIDs.emplace(&node, types.size());
        types.push_back(std::make_unique<ScalarType>(types.size(), *this, t, ar));
    }

    /**
     * Add function type
     * @param node
     * @param returnT
     * @param returnAr
     * @param typeParamIDs
     */
    void TypeInferer::insertNewFuncType(const ast::ASTNode &node,
                                        std::vector<type_id_t> typeParamIDs = {},
                                        const DataType returnT = DataType::UNKNOWN,
                                        const Arity returnAr = Arity()) {
        insertNewFuncType(node, std::move(typeParamIDs), {std::make_pair(returnT, returnAr)});
    }

    void TypeInferer::insertNewFuncType(const ast::ASTNode &node,
                                        std::vector<type_id_t> typeParamIDs = {},
                                        const std::vector<std::pair<DataType, Arity>> &returnTypes = {
                                                std::make_pair(DataType::UNKNOWN, Arity())}) {
        std::vector<type_id_t> returnTypeIds;
        for (auto t: returnTypes) {
            returnTypeIds.push_back(types.size());
            types.emplace_back(std::make_unique<ScalarType>(types.size(), *this, t.first, t.second));
        }
        insertNewFuncType(node, std::move(typeParamIDs), std::move(returnTypeIds));
    }

    void TypeInferer::insertNewFuncType(const ast::ASTNode &node,
                                        std::vector<type_id_t> typeParamIDs,
                                        const type_id_t returnTypeID) {
        insertNewFuncType(node, std::move(typeParamIDs), std::vector<type_id_t>(1, returnTypeID));
    }

    void TypeInferer::insertNewFuncType(const ast::ASTNode &node,
                                        std::vector<type_id_t> typeParamIDs,
                                        std::vector<type_id_t> returnTypeIDs) {
        typeIDs.emplace(&node, types.size());
        types.push_back(
                std::make_unique<FunctionType>(types.size(), *this, std::move(typeParamIDs), std::move(returnTypeIDs)));
    }

    type_id_t TypeInferer::get_type_id(const ast::Expression &node) {
        try {
            if (node.is_reference())
                return typeIDs.at(node.as_reference()->ref.as_expr());
            else
                return typeIDs.at(node.as_expr());
        }
        catch (std::out_of_range &) {
            throw NotInferedException();
        }
    }

    type_id_t TypeInferer::get_type_id(const ast::Statement &node) {
        try {
            if (node.is_statement_wrapper())
                return get_type_id(node.as_statement_wrapper()->expr);
            else
                return typeIDs.at(node.as_stmt());
        }
        catch (std::out_of_range &) {
            throw NotInferedException();
        }
    }

    type_id_t TypeInferer::get_type_id(const ast::ASTNode &node) {
        try {
            if (dynamic_cast<const ast::Ref *>(&node))
                return typeIDs.at(dynamic_cast<const ast::Ref *>(&node)->ref.as_expr());
            else
                return typeIDs.at(&node);
        }
        catch (std::out_of_range &) {
            throw NotInferedException();
        }
    }

    Type &TypeInferer::get_type(const ast::Expression &node) const {
        ast::IExpression *tmp = node.as_expr();
        if (node.is_reference()) // resolve reference
            tmp = node.as_reference()->ref.as_expr();
        try {
            return *types.at(typeIDs.at(tmp));
        }
        catch (std::out_of_range &) {
            throw NotInferedException();
        }
    }

    Type &TypeInferer::get_type(const ast::Statement &node) const {
        try {
            if (node.is_statement_wrapper())
                return *types.at(typeIDs.at(node.as_statement_wrapper()->expr.as_expr()));
            else
                return *types.at(typeIDs.at(node.as_stmt()));
        }
        catch (std::out_of_range &) {
            throw NotInferedException();
        }
    }

    Type &TypeInferer::get_type(const ast::ASTNode &node) const {
        try {
            if (dynamic_cast<const ast::Ref *>(&node))
                return *types.at(typeIDs.at(dynamic_cast<const ast::Ref *>(&node)->ref.as_expr()));
            else
                return *types.at(typeIDs.at(&node));
        }
        catch (std::out_of_range &) {
            throw NotInferedException();
        }
    }

    static voila::DataType get_int_type(std::intmax_t i) {
        if (i < std::numeric_limits<uint_least32_t>::max()) {
            return voila::DataType::INT32;
        } else {
            return voila::DataType::INT64;
        }
    }

    DataType TypeInferer::convert(DataType t1, DataType t2) {
        if (convertible(t1, t2)) {
            return t2;
        } else if (convertible(t2, t1)) {
            return t1;
        } else {
            throw IncompatibleTypesException();
        }
    }

    void TypeInferer::unify(ast::ASTNode &t1, ast::ASTNode &t2) {
        ast::ASTNode *tmp1 = &t1, *tmp2 = &t2;
        if (dynamic_cast<const ast::Ref *>(&t1)) {
            tmp1 = dynamic_cast<const ast::Ref *>(&t1)->ref.as_expr();
        }
        if (dynamic_cast<const ast::Ref *>(&t2)) {
            tmp2 = dynamic_cast<const ast::Ref *>(&t2)->ref.as_expr();
        }
        unify(std::vector<ast::ASTNode *>({tmp1}), tmp2);
    }

    void TypeInferer::unify(ast::ASTNode &t1, ast::Expression &t2) {
        unify(std::vector<ast::ASTNode *>({&t1}), t2.as_expr());
    }

    void TypeInferer::unify(ast::ASTNode *const t1, ast::ASTNode *const t2) {
        unify(std::vector<ast::ASTNode *>({t1}), t2);
    }

    void TypeInferer::unify(const ast::Expression &t1, const ast::Statement &t2) {
        ast::ASTNode *tmp;
        if (t2.is_statement_wrapper()) {
            tmp = t2.as_statement_wrapper()->expr.as_expr();
        } else {
            tmp = t2.as_stmt();
        }
        if (t1.is_reference())
            typeIDs.insert_or_assign(t1.as_reference()->ref.as_expr(), get_type_id(*tmp));
        else
            typeIDs.insert_or_assign(t1.as_expr(), get_type_id(*tmp));
    }

    void TypeInferer::unify(const ast::ASTNode &t1, const ast::Statement &t2) {
        if (dynamic_cast<const ast::Ref *>(&t1))
            typeIDs.insert_or_assign(dynamic_cast<const ast::Ref *>(&t1)->ref.as_expr(), get_type_id(t2));
        else
            typeIDs.insert_or_assign(&t1, get_type_id(t2));
    }

    void TypeInferer::unify(const ast::Expression &t1, const ast::Expression &t2) {
        unify(std::vector<ast::Expression>({t1}), t2);
    }

    void TypeInferer::unify(const std::vector<ast::Expression> &t1, const ast::Expression &t2) {
        const std::vector<ast::ASTNode *> tmp1 = t1 | ranges::views::transform(
                [](auto &t) -> ast::ASTNode * {
                    return t.is_reference() ? t.as_reference()->ref.as_expr() : t.as_expr();
                }) |
                                                 ranges::to_vector;
        auto *tmp2 = t2.is_reference() ? t2.as_reference()->ref.as_expr() : t2.as_expr();

        unify(tmp1, tmp2);
    }

    void TypeInferer::unify(const std::vector<ast::Expression> &t1, const ast::Statement &t2) {
        const std::vector<ast::ASTNode *> tmp1 = t1 |
                                                 ranges::views::transform(
                                                         [](auto &t) -> ast::ASTNode * {
                                                             return (t.is_reference() ? t.as_reference()->ref.as_expr()
                                                                                      : t.as_expr());
                                                         }) |
                                                 ranges::to_vector;

        const auto *tmp2 = t2.is_statement_wrapper() ?
                           dynamic_cast<ast::ASTNode *>(t2.as_statement_wrapper()->expr.as_expr()) :
                           dynamic_cast<ast::ASTNode *>(t2.as_stmt());

        unify(tmp1, tmp2);
    }

    // TODO: refactor
    void TypeInferer::unify(const std::vector<ast::ASTNode *> &t1, const ast::ASTNode *t2) {
        for (size_t i = 0; i < t1.size(); ++i) {
            auto *tmp1 = t1[i];
            if (typeIDs.contains(tmp1) && !typeIDs.contains(t2)) {
                typeIDs.emplace(t2, get_type_id(*tmp1));
            } else if (!typeIDs.contains(tmp1) && typeIDs.contains(tmp1)) {
                typeIDs.emplace(tmp1, get_type_id(*t2));
            } else if (typeIDs.contains(tmp1) && typeIDs.contains(t2)) {
                if (types[typeIDs[tmp1]]->convertible(types[typeIDs[t2]]->getTypes().at(i))) {
                    if (dynamic_cast<FunctionType *>(types[typeIDs[t2]].get())) {
                        typeIDs[tmp1] = dynamic_cast<FunctionType *>(types[typeIDs[t2]].get())->returnTypeIDs.at(i);
                    } else {
                        typeIDs[tmp1] = dynamic_cast<ScalarType *>(types[typeIDs[t2]].get())->typeID;
                    }
                } else if (dynamic_cast<FunctionType *>(types[typeIDs[t2]].get()) &&
                           types.at(dynamic_cast<FunctionType *>(types[typeIDs[t2]].get())->returnTypeIDs.at(i))
                                   ->convertible(*types[typeIDs[tmp1]])) {
                    dynamic_cast<FunctionType *>(types[typeIDs[t2]].get())->returnTypeIDs.at(i) = typeIDs[tmp1];
                } else if (dynamic_cast<ScalarType *>(types[typeIDs[t2]].get()) &&
                           types.at(dynamic_cast<ScalarType *>(types[typeIDs[t2]].get())->typeID)
                                   ->convertible(*types[typeIDs[tmp1]])) {
                    dynamic_cast<ScalarType *>(types[typeIDs[t2]].get())->typeID = typeIDs[tmp1];
                } else {
                    throw IncompatibleTypesException();
                }
            } else {
                throw NotInferedException();
            }
        }
    }

    void TypeInferer::unify(const ast::Statement &t1, const ast::Statement &t2) {
        unify(t1.as_stmt(), t2.as_stmt());
    }

    void TypeInferer::operator()(const ast::Write &write) {
        write.start.visit(*this);
        write.src.visit(*this);
        write.dest.visit(*this);

        if (!get_type(write.src).compatible(get_type(write.dest)))
            throw IncompatibleTypesException();
        if (!get_type(write.start).compatible(DataType::INT64))
            throw IncompatibleTypesException();

        // TODO: unification

        insertNewFuncType(write, {get_type_id(write.src), get_type_id(write.dest), get_type_id(write.start)},
                          DataType::VOID);
    }

    void TypeInferer::operator()(const ast::Scatter &scatter) {
        scatter.src.visit(*this);
        scatter.idxs.visit(*this);

        if (!get_type(scatter.idxs).compatible(DataType::INT64))
            throw IncompatibleTypesException();

        // TODO: unification

        insertNewFuncType(scatter, {get_type_id(scatter.idxs), get_type_id(scatter.src)},
                          get_type(scatter.src).getTypes().front(), get_type(scatter.idxs).getArities().front());
    }

    void TypeInferer::operator()(const ast::FunctionCall &call) {
        for (auto &arg: call.args) {
            arg.visit(*this);
        }

        std::vector<type_id_t> argIds;
        std::transform(
                call.args.begin(), call.args.end(),
                std::back_inserter(argIds), [&](const auto &elem) -> auto { return get_type_id(elem); });
        insertNewFuncType(call, argIds, DataType::UNKNOWN, Arity()); // TODO
    }

    void TypeInferer::operator()(const ast::Variable &var) {
        insertNewType(var);
    }

    void TypeInferer::operator()(const ast::Assign &assign) {
        for (auto &dest: assign.dests) {
            dest.visit(*this);
        }

        assign.expr.visit(*this);

        assert(assign.expr.is_function_call() || assign.expr.is_statement_wrapper());

        std::vector<type_id_t> paramIds;

        unify(assign.dests, assign.expr);

        for (const auto &dest: assign.dests) {
            paramIds.push_back(get_type_id(dest));
        }

        paramIds.push_back(get_type_id(assign.expr));
        insertNewFuncType(assign, paramIds, DataType::VOID);
    }

    void TypeInferer::operator()(const ast::Emit &emit) {
        std::vector<type_id_t> returnTypeIds;
        for (auto expr: emit.exprs) {
            expr.visit(*this);
            returnTypeIds.push_back(get_type_id(expr));
        }

        insertNewFuncType(emit, {}, returnTypeIds);
    }

    void TypeInferer::operator()(const ast::Loop &loop) {
        loop.pred.visit(*this);

        for (auto &stmt: loop.stms) {
            stmt.visit(*this);
        }

        if (!get_type(loop.pred).compatible(DataType::BOOL)) {
            throw IncompatibleTypesException();
        }
    }

    void TypeInferer::operator()(const ast::Hash &hash) {
        // TODO: check same arities
        for (auto &elem: hash.items)
            elem.visit(*this);

        std::vector<type_id_t> type_ids;
        for (const auto &elem: hash.items)
            type_ids.push_back(get_type_id(elem));

        assert(get_type(hash.items.front()).getArities().size() == 1);
        insertNewFuncType(hash, type_ids, DataType::INT64, get_type(hash.items.front()).getArities().front());
    }

    void TypeInferer::operator()(const ast::Arithmetic &arithmetic) {
        arithmetic.lhs.visit(*this);
        arithmetic.rhs.visit(*this);

        const auto &left_type = get_type(arithmetic.lhs);
        const auto &right_type = get_type(arithmetic.rhs);
        if (left_type.getArities() != right_type.getArities()) {
            throw NonMatchingArityException();
        }
        if (!left_type.compatible(DataType::NUMERIC) || !right_type.compatible(DataType::NUMERIC)) {
            throw IncompatibleTypesException();
        }

        if (left_type.getTypes() != right_type.getTypes()) {
            unify(arithmetic.lhs, arithmetic.rhs);
        }

        assert(get_type(arithmetic.lhs).getTypes().size() == 1);
        assert(left_type.getArities().size());

        insertNewFuncType(arithmetic, {get_type_id(arithmetic.lhs), get_type_id(arithmetic.rhs)},
                          get_type(arithmetic.lhs).getTypes().front(), left_type.getArities().front());
    }

    void TypeInferer::operator()(const ast::IntConst &aConst) {
        assert(!typeIDs.contains(&aConst));
        insertNewType(aConst, get_int_type(aConst.val));
    }

    void TypeInferer::operator()(const ast::BooleanConst &aConst) {
        assert(!typeIDs.contains(&aConst));
        insertNewType(aConst, DataType::BOOL);
    }

    void TypeInferer::operator()(const ast::FltConst &aConst) {
        assert(!typeIDs.contains(&aConst));
        insertNewType(aConst, DataType::DBL);
    }

    void TypeInferer::operator()(const ast::StrConst &aConst) {
        assert(!typeIDs.contains(&aConst));
        insertNewType(aConst, DataType::STRING);
    }

    void TypeInferer::operator()(const ast::Read &read) {
        read.column.visit(*this);
        read.idx.visit(*this);

        if (!get_type(read.idx).compatible(DataType::INT64))
            throw IncompatibleTypesException();
        assert(get_type(read.column).getTypes().size() == 1);
        if (get_type(read.column).getTypes().front() == DataType::VOID)
            throw IncompatibleTypesException();

        insertNewFuncType(read, {get_type_id(read.column), get_type_id(read.idx)}, DataType::VOID);
    }

    void TypeInferer::operator()(const ast::Gather &gather) {
        gather.column.visit(*this);
        gather.idxs.visit(*this);

        if (!get_type(gather.idxs).compatible(DataType::INT64))
            throw IncompatibleTypesException();
        assert(get_type(gather.column).getTypes().size() == 1);
        if (get_type(gather.column).getTypes().front() == DataType::VOID)
            throw IncompatibleTypesException();

        insertNewFuncType(gather, {get_type_id(gather.column), get_type_id(gather.idxs)},
                          get_type(gather.column).getTypes().front(), get_type(gather.idxs).getArities().front());
    }

    void TypeInferer::operator()(const ast::Ref &) {
        // do not infer type, just act as a wrapper around variable
    }

    void TypeInferer::operator()(const ast::TupleGet &) {
        // TODO: check expr list
        // insertNewType(get, get_type(get.expr));
    }

    void TypeInferer::operator()(const ast::TupleCreate &create) {
        // TODO
        ASTVisitor::operator()(create);
    }

    void TypeInferer::operator()(const ast::Fun &fun) {
        // TODO: clear infered types at start of new function?
        for (auto &arg: fun.args) // infer function args
        {
            arg.visit(*this);
        }
        for (auto &stmt: fun.body) // infer body types
        {
            stmt.visit(*this);
        }
        std::vector<type_id_t> argIds;
        std::transform(
                fun.args.begin(), fun.args.end(),
                std::back_inserter(argIds), [&](const auto &elem) -> auto { return get_type_id(elem); });
        if (fun.result.has_value())
            insertNewFuncType(fun, argIds, get_type_id(*fun.result));
        else
            insertNewFuncType(fun, argIds, DataType::VOID);
    }

    void TypeInferer::operator()(const ast::Main &main) {
        // TODO: clear infered types at start of new function?
        // do not infer function args, they have to be specified by user before further inference

        for (auto &stmt: main.body) // infer body types
        {
            stmt.visit(*this);
        }

        std::vector<type_id_t> argIds;
        std::transform(
                main.args.begin(), main.args.end(),
                std::back_inserter(argIds), [&](const auto &elem) -> auto { return get_type_id(elem); });
        if (main.result.has_value()) {
            insertNewFuncType(main, argIds, get_type_id(*main.result));
        } else {
            insertNewFuncType(main, argIds, DataType::VOID);
        }
    }

    void TypeInferer::operator()(const ast::Selection &selection) {
        selection.pred.visit(*this);
        selection.param.visit(*this);

        auto &type = get_type(*selection.param.as_expr());

        if (!get_type(selection.pred).convertible(DataType::BOOL)) {
            throw IncompatibleTypesException();
        } else {
            dynamic_cast<ScalarType &>(get_type(selection.pred)).t = DataType::BOOL;
        }

        assert(!type.getTypes().empty());
        insertNewFuncType(selection, {get_type_id(selection.param), get_type_id(selection.param)},
                          type.getTypes().front());
    }

    void TypeInferer::operator()(const ast::Lookup &lookup) {
        ranges::for_each(
                lookup.values, [&](auto &t) -> auto { t.visit(*this); });
        ranges::for_each(
                lookup.tables, [&](auto &t) -> auto { t.visit(*this); });
        lookup.hashes.visit(*this);

        std::vector<type_id_t> paramTypeIds =
                ranges::views::concat(
                        lookup.values | ranges::views::transform([&](auto &val) { return get_type_id(val); }),
                        lookup.tables | ranges::views::transform([&](auto &val) { return get_type_id(val); })) |
                ranges::to_vector;

        paramTypeIds.push_back(get_type_id(lookup.hashes));

        insertNewFuncType(lookup, paramTypeIds, DataType::INT64, get_type(lookup.hashes).getArities().front());
    }

    void TypeInferer::operator()(const ast::Insert &insert) {
        insert.keys.visit(*this);
        for (auto &val: insert.values)
            val.visit(*this);

        // TODO: return multiple output tables
        assert(get_type(insert.keys).getArities().size() == 1);
        std::vector<type_id_t> paramTypeIds;
        paramTypeIds.push_back(get_type_id(insert.keys));

        ranges::copy(insert.values | ranges::views::transform([&](auto &val) { return get_type_id(val); }) |
                     ranges::to_vector, ranges::back_inserter(paramTypeIds));

        const std::vector<std::pair<DataType, Arity>> returnTypes =
                insert.values | ranges::views::transform([&](auto &val) -> std::pair<DataType, Arity> {
                    return std::make_pair(get_type(val).getTypes().front(), get_type(insert.keys).getArities().front());
                }) | ranges::to_vector;

        insertNewFuncType(insert, paramTypeIds, returnTypes);
    }

    void TypeInferer::operator()(const ast::Predicate &pred) {
        pred.expr.visit(*this);

        auto &type = get_type(*pred.expr.as_expr());

        if (!get_type(pred.expr).convertible(DataType::BOOL)) {
            throw IncompatibleTypesException();
        } else {
            dynamic_cast<ScalarType &>(get_type(pred.expr)).t = DataType::BOOL;
        }

        assert(type.getTypes().size() == 1);
        insertNewFuncType(pred, {get_type_id(pred.expr), get_type_id(pred.expr)}, type.getTypes().front());
    }

    void TypeInferer::operator()(const ast::Comparison &comparison) {
        comparison.lhs.visit(*this);
        comparison.rhs.visit(*this);

        auto &left_type = get_type(*comparison.lhs.as_expr());
        auto &right_type = get_type(*comparison.rhs.as_expr());
        if (left_type.getArities().size() != 1) {
            throw NonMatchingArityException();
        }
        if (right_type.getArities().size() != 1) {
            throw NonMatchingArityException();
        }
        auto &leftArities = left_type.getArities().front().get();
        auto &rightArities = right_type.getArities().front().get();

        // TODO: insert for all binary preds
        if (leftArities.is_undef() xor rightArities.is_undef()) {
            if (leftArities.is_undef()) {
                leftArities = rightArities;
            } else {
                rightArities = leftArities;
            }
        }
        if (const_cast<const Type &>(left_type).getArities() !=
            const_cast<const Type &>(right_type).getArities()) {
            throw NonMatchingArityException();
        }
        /*        if (!convertible(left_type.t, DataType::BOOL) || !convertible(right_type.t, DataType::BOOL))
                {
                    throw IncompatibleTypesException();
                }*/

        if (left_type.getTypes() != right_type.getTypes()) {
            unify(comparison.lhs, comparison.rhs);
        }

        insertNewFuncType(comparison, {get_type_id(comparison.lhs), get_type_id(comparison.rhs)}, DataType::BOOL,
                          Arity(get_type(comparison.lhs).getArities().front()));
    }

    void TypeInferer::operator()(const ast::Add &var) {
        TypeInferer::operator()(static_cast<const ast::Arithmetic &>(var));
    }

    void TypeInferer::operator()(const ast::Sub &sub) {
        TypeInferer::operator()(static_cast<const ast::Arithmetic &>(sub));
    }

    void TypeInferer::operator()(const ast::Mul &mul) {
        TypeInferer::operator()(static_cast<const ast::Arithmetic &>(mul));
    }

    void TypeInferer::operator()(const ast::Div &div1) {
        TypeInferer::operator()(static_cast<const ast::Arithmetic &>(div1));
    }

    void TypeInferer::operator()(const ast::Mod &mod) {
        TypeInferer::operator()(static_cast<const ast::Arithmetic &>(mod));
    }

    void TypeInferer::operator()(const ast::AggrSum &sum) {
        sum.src.visit(*this);
        if (sum.groups) {
            sum.groups->visit(*this);
            insertNewFuncType(sum, {get_type_id(sum.src), get_type_id(*sum.groups)},
                              get_type(sum.src).getTypes().front() == DataType::DBL ? DataType::DBL : DataType::INT64);
        } else {
            insertNewFuncType(sum, {get_type_id(sum.src)}, get_type(sum.src).getTypes().front(), Arity(1));
        }
    }

    void TypeInferer::operator()(const ast::AggrCnt &cnt) {
        cnt.src.visit(*this);
        if (cnt.groups) {
            cnt.groups->visit(*this);
            insertNewFuncType(cnt, {get_type_id(cnt.src), get_type_id(*cnt.groups)}, DataType::INT64);
        } else {
            insertNewFuncType(cnt, {get_type_id(cnt.src)}, DataType::INT64, Arity(1));
        }
    }

    void TypeInferer::operator()(const ast::AggrMin &aggrMin) {
        aggrMin.src.visit(*this);
        if (aggrMin.groups) {
            aggrMin.groups->visit(*this);
            insertNewFuncType(aggrMin, {get_type_id(aggrMin.src), get_type_id(*aggrMin.groups)},
                              get_type(aggrMin.src).getTypes().front());
        } else {
            insertNewFuncType(aggrMin, {get_type_id(aggrMin.src)}, get_type(aggrMin.src).getTypes().front(), Arity(1));
        }
    }

    void TypeInferer::operator()(const ast::AggrMax &aggrMax) {
        aggrMax.src.visit(*this);
        if (aggrMax.groups) {
            aggrMax.groups->visit(*this);
            insertNewFuncType(aggrMax, {get_type_id(aggrMax.src), get_type_id(*aggrMax.groups)},
                              get_type(aggrMax.src).getTypes().front());
        } else {
            insertNewFuncType(aggrMax, {get_type_id(aggrMax.src)}, get_type(aggrMax.src).getTypes().front(), Arity(1));
        }
    }

    void TypeInferer::operator()(const ast::AggrAvg &avg) {
        avg.src.visit(*this);
        if (avg.groups) {
            avg.groups->visit(*this);
            insertNewFuncType(avg, {get_type_id(avg.src), get_type_id(*avg.groups)}, DataType::DBL);
        } else {
            insertNewFuncType(avg, {get_type_id(avg.src)}, DataType::DBL, Arity(1));
        }
    }

    void TypeInferer::operator()(const ast::Eq &eq) {
        TypeInferer::operator()(static_cast<const ast::Comparison &>(eq));
    }

    void TypeInferer::operator()(const ast::Neq &neq) {
        TypeInferer::operator()(static_cast<const ast::Comparison &>(neq));
    }

    void TypeInferer::operator()(const ast::Le &le) {
        TypeInferer::operator()(static_cast<const ast::Comparison &>(le));
    }

    void TypeInferer::operator()(const ast::Ge &ge) {
        TypeInferer::operator()(static_cast<const ast::Comparison &>(ge));
    }

    void TypeInferer::operator()(const ast::Leq &leq) {
        TypeInferer::operator()(static_cast<const ast::Comparison &>(leq));
    }

    void TypeInferer::operator()(const ast::Geq &geq) {
        TypeInferer::operator()(static_cast<const ast::Comparison &>(geq));
    }

    void TypeInferer::operator()(const ast::And &anAnd) {
        anAnd.lhs.visit(*this);
        anAnd.rhs.visit(*this);

        const auto &left_type = get_type(*anAnd.lhs.as_expr());
        const auto &right_type = get_type(*anAnd.rhs.as_expr());
        if (left_type.getArities() != right_type.getArities()) {
            throw NonMatchingArityException();
        }
        if (!left_type.convertible(DataType::BOOL) || !right_type.convertible(DataType::BOOL)) {
            throw IncompatibleTypesException();
        }

        if (left_type.getTypes() != right_type.getTypes()) {
            unify(anAnd.lhs, anAnd.rhs);
        }

        insertNewFuncType(anAnd, {get_type_id(anAnd.lhs), get_type_id(anAnd.rhs)}, DataType::BOOL,
                          Arity(get_type(anAnd.lhs).getArities().front()));
    }

    void TypeInferer::operator()(const ast::Or &anOr) {
        anOr.lhs.visit(*this);
        anOr.rhs.visit(*this);

        const auto &left_type = get_type(*anOr.lhs.as_expr());
        const auto &right_type = get_type(*anOr.rhs.as_expr());
        if (left_type.getArities() != right_type.getArities()) {
            throw NonMatchingArityException();
        }
        if (!left_type.convertible(DataType::BOOL) || !right_type.convertible(DataType::BOOL)) {
            throw IncompatibleTypesException();
        }

        if (left_type.getTypes() != right_type.getTypes()) {
            unify(anOr.lhs, anOr.rhs);
        }

        insertNewFuncType(anOr, {get_type_id(anOr.lhs), get_type_id(anOr.rhs)}, DataType::BOOL,
                          Arity(get_type(anOr.lhs).getArities().front()));
    }

    void TypeInferer::operator()(const ast::Not &aNot) {
        aNot.param.visit(*this);

        auto &type = get_type(*aNot.param.as_expr());

        if (!type.convertible(DataType::BOOL)) {
            throw IncompatibleTypesException();
        } else {
            dynamic_cast<ScalarType &>(type).t = DataType::BOOL;
        }

        insertNewFuncType(aNot, {get_type_id(aNot.param)}, DataType::BOOL,
                          Arity(get_type(aNot.param).getArities().front()));
    }

    void TypeInferer::operator()(const ast::StatementWrapper &wrapper) {
        wrapper.expr.visit(*this);
    }

    void TypeInferer::set_arity(const ast::ASTNode *const node, const size_t ar) {
        if (dynamic_cast<ScalarType *>(types.at(typeIDs.at(node)).get())) {
            dynamic_cast<ScalarType *>(types.at(typeIDs.at(node)).get())->ar = Arity(ar);
        } else // what when returnTypes > 1 elem
        {
            dynamic_cast<ScalarType *>(
                    types.at(dynamic_cast<FunctionType *>(types.at(
                            typeIDs.at(node)).get())->returnTypeIDs.front()).get())
                    ->ar = Arity(ar);
        }
    }

    void TypeInferer::set_type(const ast::ASTNode *const node, const DataType type) {
        if (dynamic_cast<ScalarType *>(&get_type(*node))) {
            dynamic_cast<ScalarType &>(get_type(*node)).t = type;
        } else {
            // TODO:what when returnTypes > 1 elem
            dynamic_cast<ScalarType *>(
                    types.at(dynamic_cast<FunctionType *>(types.at(
                            typeIDs.at(node)).get())->returnTypeIDs.front()).get())
                    ->t = type;
        }
    }

} // namespace voila
#include "MLIRGeneratorImpl.hpp"

#include "MlirGenerationException.hpp"
#include "NotImplementedException.hpp"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"

#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/TypeUtilities.h>

namespace voila::mlir
{
    using namespace ast;
    using namespace ::mlir;
    using namespace ::mlir::voila;
    using ::llvm::ScopedHashTableScope;
    using ::llvm::SmallVector;
    using ::llvm::StringRef;
    using ::mlir::Value;
    using ::mlir::ValueRange;

    void MLIRGeneratorImpl::operator()(const AggrSum &sum)
    {
        auto location = loc(sum.get_location());
        auto expr = std::get<Value>(visitor_gen(sum.src));

        ::mlir::Type resType;
        if (sum.groups)
        {
            if (getElementTypeOrSelf(expr).isIntOrIndex())
            {
                resType = RankedTensorType::get(-1, builder.getI64Type());
            }
            else if (getElementTypeOrSelf(expr).isIntOrFloat())
            {
                resType = RankedTensorType::get(-1, builder.getF64Type());
            }
            else
            {
                throw MLIRGenerationException();
            }
            auto idxs = std::get<Value>(visitor_gen(*sum.groups));
            result = builder.create<SumOp>(location, resType, expr, idxs);
        }
        else
        {
            if (getElementTypeOrSelf(expr).isIntOrIndex())
            {
                resType = builder.getI64Type();
            }
            else if (getElementTypeOrSelf(expr).isIntOrFloat())
            {
                resType = builder.getF64Type();
            }
            else
            {
                throw MLIRGenerationException();
            }

            result = builder.create<SumOp>(location, resType, expr);
        }
    }

    void MLIRGeneratorImpl::operator()(const AggrCnt &cnt)
    {
        auto location = loc(cnt.get_location());

        mlir::Value expr = std::get<Value>(visitor_gen(cnt.src));

        if (cnt.groups)
        {
            auto idxs = std::get<Value>(visitor_gen(*cnt.groups));
            result =
                builder.create<CountOp>(location, RankedTensorType::get(-1, builder.getI64Type()), expr, idxs);
        }
        else
        {
            result = builder.create<CountOp>(location, builder.getI64Type(), expr);
        }
    }

    void MLIRGeneratorImpl::operator()(const AggrMin &min)
    {
        auto location = loc(min.get_location());

        mlir::Value expr = std::get<Value>(visitor_gen(min.src));
        ::mlir::Type resType;
        if (min.groups)
        {
            if (getElementTypeOrSelf(expr).isIntOrIndex())
            {
                resType = RankedTensorType::get(-1, builder.getI64Type());
            }
            else if (getElementTypeOrSelf(expr).isIntOrFloat())
            {
                resType = RankedTensorType::get(-1, builder.getF64Type());
            }
            else
            {
                throw MLIRGenerationException();
            }
            auto idxs = std::get<Value>(visitor_gen(*min.groups));

            result = builder.create<MinOp>(location, resType, expr, idxs);
        }
        else
        {
            if (getElementTypeOrSelf(expr).isIntOrIndex())
            {
                resType = builder.getI64Type();
            }
            else if (getElementTypeOrSelf(expr).isIntOrFloat())
            {
                resType = builder.getF64Type();
            }
            else
            {
                throw MLIRGenerationException();
            }

            result = builder.create<MinOp>(location, resType, expr);
        }
    }

    void MLIRGeneratorImpl::operator()(const AggrMax &max)
    {
        auto location = loc(max.get_location());

        mlir::Value expr = std::get<Value>(visitor_gen(max.src));
        ::mlir::Type resType;
        if (max.groups)
        {
            if (getElementTypeOrSelf(expr).isIntOrIndex())
            {
                resType = RankedTensorType::get(-1, builder.getI64Type());
            }
            else if (getElementTypeOrSelf(expr).isIntOrFloat())
            {
                resType = RankedTensorType::get(-1, builder.getF64Type());
            }
            else
            {
                throw MLIRGenerationException();
            }
            auto idxs = std::get<Value>(visitor_gen(*max.groups));

            result = builder.create<MaxOp>(location, resType, expr, idxs);
        }
        else
        {
            if (getElementTypeOrSelf(expr).isIntOrIndex())
            {
                resType = builder.getI64Type();
            }
            else if (getElementTypeOrSelf(expr).isIntOrFloat())
            {
                resType = builder.getF64Type();
            }
            else
            {
                throw MLIRGenerationException();
            }

            result = builder.create<MaxOp>(location, resType, expr);
        }
    }

    // TODO
    void MLIRGeneratorImpl::operator()(const AggrAvg &avg)
    {
        auto location = loc(avg.get_location());
        mlir::Value expr = std::get<Value>(visitor_gen(avg.src));

        if (avg.groups)
        {
            auto idxs = std::get<Value>(visitor_gen(*avg.groups));
            result =
                builder.create<AvgOp>(location, RankedTensorType::get(-1, builder.getF64Type()), expr, idxs);
        }
        else
        {
            result = builder.create<AvgOp>(location, builder.getF64Type(), expr);
        }
    }

    void MLIRGeneratorImpl::operator()(const Write &write)
    {
        ASTVisitor::operator()(write);
    }

    void MLIRGeneratorImpl::operator()(const Scatter &scatter)
    {
        auto location = loc(scatter.get_location());
        auto col = std::get<Value>(visitor_gen(scatter.src));
        auto idx = std::get<Value>(visitor_gen(scatter.idxs));
        result = builder.create<ScatterOp>(location, col.getType(), idx, col, Value());
    }

    void MLIRGeneratorImpl::operator()(const FunctionCall &call)
    {
        auto location = loc(call.get_location());
        SmallVector<Value> operands;
        for (auto &expr : call.args)
        {
            auto arg = visitor_gen(expr);
            operands.push_back(std::get<Value>(arg));
        }
        // TODO: allow more than only a single return type
        result = builder.create<GenericCallOp>(location, call.fun, operands,
                                               funcTable.lookup(call.fun).getFunctionType().getResult(0));
    }

    void MLIRGeneratorImpl::operator()(const Assign &assign)
    {
        auto res = visitor_gen(assign.expr);
        result = res;
        ValueRange val;
        if (std::holds_alternative<Value>(res))
            val = {std::get<Value>(res)};
        else if ((std::holds_alternative<SmallVector<Value>>(res)))
            val = std::get<SmallVector<Value>>(res);

        // assign value to variable
        for (size_t i = 0; i < assign.dests.size(); ++i)

        {
            auto dest = assign.dests[i];
            auto value = val[i];

            symbolTable.insert((dest.is_variable() ? dest.as_variable() : dest.as_reference()->ref.as_variable())->var,
                               value);
        }
    }

    void MLIRGeneratorImpl::operator()(const Emit &emit)
    {
        auto location = loc(emit.get_location());

        // 'return' takes an optional expression, handle that case here.
        auto exprs = std::get<SmallVector<Value>>(visitor_gen(emit.exprs));

        // Otherwise, this return operation has zero operands.
        builder.create<EmitOp>(location, exprs);
        result = ::mlir::success();
    }

    void MLIRGeneratorImpl::operator()(const Loop &loop)
    {
        auto location = loc(loop.get_location());
        mlir::Value cond = std::get<Value>(visitor_gen(loop.pred));

        auto voilaLoop = builder.create<LoopOp>(location, builder.getI1Type(), cond);
        auto &bodyRegion = voilaLoop.body();
        bodyRegion.push_back(new ::mlir::Block());
        ::mlir::Block &bodyBlock = bodyRegion.front();
        ::mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&bodyBlock);
        for (const auto &elem : loop.stms)
        {
            visitor_gen(elem);
        }

        result = ::mlir::success();
    }

    void MLIRGeneratorImpl::operator()(const StatementWrapper &wrapper)
    {
        // forward to expr
        wrapper.expr.visit(*this);
    }

    void MLIRGeneratorImpl::operator()(const Add &add)
    {
        auto location = loc(add.get_location());
        auto lhs = visitor_gen(add.lhs);
        auto rhs = visitor_gen(add.rhs);
        auto resType = getTypes(add);
        assert(resType.size() == 1);
        result = builder.create<AddOp>(location, resType, std::get<Value>(lhs), std::get<Value>(rhs));
    }

    void MLIRGeneratorImpl::operator()(const Sub &sub)
    {
        auto location = loc(sub.get_location());
        auto lhs = visitor_gen(sub.lhs);
        auto rhs = visitor_gen(sub.rhs);
        auto resType = getTypes(sub);
        assert(resType.size() == 1);
        result = builder.create<SubOp>(location, resType, std::get<Value>(lhs), std::get<Value>(rhs));
    }

    void MLIRGeneratorImpl::operator()(const Mul &mul)
    {
        auto location = loc(mul.get_location());
        auto lhs = visitor_gen(mul.lhs);
        auto rhs = visitor_gen(mul.rhs);
        auto resType = getTypes(mul);
        assert(resType.size() == 1);
        result = builder.create<MulOp>(location, resType, std::get<Value>(lhs), std::get<Value>(rhs));
    }

    void MLIRGeneratorImpl::operator()(const Div &div)
    {
        auto location = loc(div.get_location());
        auto lhs = visitor_gen(div.lhs);
        auto rhs = visitor_gen(div.rhs);

        auto resType = getTypes(div);
        assert(resType.size() == 1);
        result = builder.create<DivOp>(location, resType, std::get<Value>(lhs), std::get<Value>(rhs));
    }

    void MLIRGeneratorImpl::operator()(const Mod &mod)
    {
        auto location = loc(mod.get_location());
        auto lhs = visitor_gen(mod.lhs);
        auto rhs = visitor_gen(mod.rhs);
        auto resType = getTypes(mod);
        assert(resType.size() == 1);
        result = builder.create<ModOp>(location, resType, std::get<Value>(lhs), std::get<Value>(rhs));
    }

    void MLIRGeneratorImpl::operator()(const Eq &eq)
    {
        result = getCmpOp<EqOp>(eq);
    }

    void MLIRGeneratorImpl::operator()(const Neq &neq)
    {
        result = getCmpOp<NeqOp>(neq);
    }

    void MLIRGeneratorImpl::operator()(const Le &le)
    {
        result = getCmpOp<LeOp>(le);
    }

    void MLIRGeneratorImpl::operator()(const Ge &ge)
    {
        result = getCmpOp<GeOp>(ge);
    }

    void MLIRGeneratorImpl::operator()(const Leq &leq)
    {
        result = getCmpOp<LeqOp>(leq);
    }

    void MLIRGeneratorImpl::operator()(const Geq &geq)
    {
        result = getCmpOp<GeqOp>(geq);
    }

    void MLIRGeneratorImpl::operator()(const And &anAnd)
    {
        auto location = loc(anAnd.get_location());
        auto lhs = std::get<::mlir::Value>(visitor_gen(anAnd.lhs));
        auto rhs = std::get<::mlir::Value>(visitor_gen(anAnd.rhs));
        if (lhs.getType().isa<::mlir::TensorType>() || rhs.getType().isa<::mlir::TensorType>())
        {
            ::mlir::ArrayRef<int64_t> shape;
            shape = getShape(lhs, rhs);

            result = builder.create<AndOp>(location, ::mlir::RankedTensorType::get(shape, builder.getI1Type()), lhs,
                                           rhs);
        }
        else
            result = builder.create<AndOp>(location, builder.getI1Type(), lhs, rhs);
    }

    void MLIRGeneratorImpl::operator()(const Or &anOr)
    {
        auto location = loc(anOr.get_location());
        auto lhs = std::get<Value>(visitor_gen(anOr.lhs));
        auto rhs = std::get<Value>(visitor_gen(anOr.rhs));
        result =
            builder.create<OrOp>(location, ::mlir::RankedTensorType::get(-1, builder.getI1Type()), lhs, rhs);
    }

    void MLIRGeneratorImpl::operator()(const Not &aNot)
    {
        auto location = loc(aNot.get_location());
        auto param = visitor_gen(aNot.param);

        result = builder.create<NotOp>(location, ::mlir::RankedTensorType::get(-1, builder.getI1Type()),
                                       std::get<Value>(param));
    }

    void MLIRGeneratorImpl::operator()(const Hash &hash)
    {
        auto location = loc(hash.get_location());
        auto params = std::get<SmallVector<Value>>(visitor_gen(hash.items));
        auto retType = ::mlir::RankedTensorType::get(params.front().getType().dyn_cast<ShapedType>().getShape(),
                                                     builder.getI64Type());
        result = builder.create<HashOp>(location, retType, params);
    }

    void MLIRGeneratorImpl::operator()(const IntConst &intConst)
    {
        result = builder.create<IntConstOp>(loc(intConst.get_location()), getScalarType(intConst),
                                                           intConst.val);
    }

    void MLIRGeneratorImpl::operator()(const BooleanConst &booleanConst)
    {
        result = builder.create<BoolConstOp>(loc(booleanConst.get_location()), builder.getI1Type(),
                                                            booleanConst.val);
    }

    void MLIRGeneratorImpl::operator()(const FltConst &fltConst)
    {
        result = builder.create<FltConstOp>(loc(fltConst.get_location()),
                                                           builder.getF64FloatAttr(fltConst.val));
    }

    void MLIRGeneratorImpl::operator()(const StrConst &)
    {
        throw NotImplementedException();
    }

    void MLIRGeneratorImpl::operator()(const Predicate &pred)
    {
        result = std::get<Value>(visitor_gen(pred.expr));
    }

    void MLIRGeneratorImpl::operator()(const Read &read)
    {
        auto location = loc(read.get_location());
        auto col = std::get<Value>(visitor_gen(read.column));
        auto idx = std::get<Value>(visitor_gen(read.idx));

        result = builder.create<ReadOp>(location, col.getType(), col, idx, Value());
    }

    void MLIRGeneratorImpl::operator()(const Gather &gather)
    {
        auto location = loc(gather.get_location());
        auto col = std::get<Value>(visitor_gen(gather.column));
        auto idx = std::get<Value>(visitor_gen(gather.idxs));
        result = builder.create<GatherOp>(
            location, RankedTensorType::get(idx.getType().dyn_cast<TensorType>().getShape(), getElementTypeOrSelf(col)),
            col, idx, Value());
    }

    void MLIRGeneratorImpl::operator()(const Ref &param)
    {
        if (symbolTable.count(param.ref.as_variable()->var))
        {
            auto variable = symbolTable.lookup(param.ref.as_variable()->var);
            result = variable;
            return;
        }
        throw MLIRGenerationException();
    }

    void MLIRGeneratorImpl::operator()(const TupleGet &)
    {
        throw NotImplementedException();
    }

    void MLIRGeneratorImpl::operator()(const TupleCreate &)
    {
        throw NotImplementedException();
    }

    void MLIRGeneratorImpl::operator()(const Fun &fun)
    {
        ScopedHashTableScope<StringRef, Value> var_scope(symbolTable);

        auto location = loc(fun.loc);

        // generic function, the return type will be inferred later.
        // Arguments type are uniformly unranked tensors.

        llvm::SmallVector<::mlir::Type> arg_types;
        for (const auto &t : fun.args)
        {
            auto types = getTypes(*t.as_expr());
            arg_types.insert(arg_types.end(), types.begin(), types.end());
        }

        auto func_type = builder.getFunctionType(arg_types, llvm::None);
        auto function = ::mlir::func::FuncOp::create(location, fun.name, func_type);
        funcTable.insert(std::make_pair(fun.name, function));
        assert(function);

        auto &entryBlock = *function.addEntryBlock();
        auto protoArgs = fun.args;

        // Declare all the function arguments in the symbol table.
        for (const auto nameValue : llvm::zip(protoArgs, entryBlock.getArguments()))
        {
            declare(std::get<0>(nameValue).as_variable()->var, std::get<1>(nameValue));
        }

        builder.setInsertionPointToStart(&entryBlock);

        // Emit the body of the function.
        mlirGenBody(fun.body);

        EmitOp emitOp;
        if (!entryBlock.empty())
            emitOp = dyn_cast<EmitOp>(entryBlock.back());
        if (!emitOp)
        {
            builder.create<EmitOp>(loc(fun.get_location()));
        }
        else if (emitOp.hasOperand())
        {
            // Otherwise, if this return operation has an operand then add a result to
            // the function.
            // TODO: get emit type
            function.setType(
                builder.getFunctionType(function.getFunctionType().getInputs(), getTypes(*(*fun.result).as_stmt())));
        }

        result = function;
    }

    void MLIRGeneratorImpl::operator()(const Main &main)
    {
        // TODO: can we just slice main to fun, or do we have to consider some special properties of main?
        operator()(static_cast<Fun>(main));
    }

    void MLIRGeneratorImpl::operator()(const Selection &selection)
    {
        auto location = loc(selection.get_location());
        auto values = std::get<Value>(visitor_gen(selection.param));
        auto pred = std::get<Value>(visitor_gen(selection.pred));
        assert(pred.getType().isa<::mlir::TensorType>());

        result = builder.create<::mlir::voila::SelectOp>(
            location, ::mlir::RankedTensorType::get(-1, getElementTypeOrSelf(values)), values, pred);
    }

    void MLIRGeneratorImpl::operator()(const Lookup &lookup)
    {
        auto location = loc(lookup.get_location());
        auto tables = std::get<SmallVector<Value>>(visitor_gen(lookup.tables));
        auto hashes = std::get<Value>(visitor_gen(lookup.hashes));
        auto values = std::get<SmallVector<Value>>(visitor_gen(lookup.values));

        result = builder.create<LookupOp>(
            location,
            ::mlir::RankedTensorType::get(hashes.getType().dyn_cast<::mlir::TensorType>().getShape(),
                                          builder.getIndexType()),
            values, tables, hashes);
    }

    void MLIRGeneratorImpl::operator()(const Insert &insert)
    {
        auto location = loc(insert.get_location());
        auto table = std::get<Value>(visitor_gen(insert.keys));
        auto data = std::get<SmallVector<Value>>(visitor_gen(insert.values));

        ::mlir::SmallVector<::mlir::Type> retTypes;
        for (auto val : data)
        {
            retTypes.push_back(::mlir::RankedTensorType::get({-1}, getElementTypeOrSelf(val)));
        }
        auto insertOp = builder.create<InsertOp>(location, retTypes, table, data);
        result = insertOp.hashtables();
    }

    void MLIRGeneratorImpl::operator()(const Variable &variable)
    {
        // Register the value in the symbol table.
        // TODO: SSA
        declare(variable.var, nullptr);
        result = symbolTable.lookup(variable.var);
    }

    std::vector<::mlir::Type> MLIRGeneratorImpl::getTypes(const ASTNode &node)
    {
        const auto &astType = inferer.get_type(node);
        std::vector<::mlir::Type> types;
        if (dynamic_cast<const ScalarType *>(&astType))
        {
            types.push_back(convert(astType));
        }
        else
        {
            for (auto tid : dynamic_cast<const FunctionType *>(&astType)->returnTypeIDs)
            {
                types.push_back(convert(*inferer.types.at(tid)));
            }
        }
        return types;
    }

    ::mlir::Type MLIRGeneratorImpl::convert(const ::voila::Type &t)
    {
        assert(dynamic_cast<const ScalarType *>(&t) ||
               dynamic_cast<const FunctionType *>(&t)->returnTypeIDs.size() == 1);
        switch (t.getTypes().front())
        {
            case DataType::BOOL:
                if (t.getArities().front().is_undef())
                {
                    return ::mlir::RankedTensorType::get(-1, builder.getI1Type());
                }
                else if (t.getArities().front().get_size() <= 1)
                {
                    return builder.getI1Type();
                }
                else
                {
                    return ::mlir::RankedTensorType::get(t.getArities().front().get_size(), builder.getI1Type());
                }
            case DataType::INT32:
                if (t.getArities().front().is_undef())
                {
                    return ::mlir::RankedTensorType::get(-1, builder.getI32Type());
                }
                else if (t.getArities().front().get_size() <= 1)
                {
                    return builder.getI32Type();
                }
                else
                {
                    return ::mlir::RankedTensorType::get(t.getArities().front().get_size(), builder.getI32Type());
                }
            case DataType::INT64:
                if (t.getArities().front().is_undef())
                {
                    return ::mlir::RankedTensorType::get(-1, builder.getI64Type());
                }
                else if (t.getArities().front().get_size() <= 1)
                {
                    return builder.getI64Type();
                }
                else
                {
                    return ::mlir::RankedTensorType::get(t.getArities().front().get_size(), builder.getI64Type());
                }
            case DataType::DBL:
                if (t.getArities().front().is_undef())
                {
                    return ::mlir::RankedTensorType::get(-1, builder.getF64Type());
                }
                else if (t.getArities().front().get_size() <= 1)
                {
                    return builder.getF64Type();
                }
                else
                {
                    return ::mlir::RankedTensorType::get(t.getArities().front().get_size(), builder.getF64Type());
                }
            default:
                // TODO
                throw NotImplementedException();
        }
    }

    ::mlir::Type MLIRGeneratorImpl::getScalarType(const ASTNode &node)
    {
        const auto &astType = inferer.get_type(node);
        return scalarConvert(astType);
    }

    ::mlir::Type MLIRGeneratorImpl::scalarConvert(const ::voila::Type &t)
    {
        assert(dynamic_cast<const ScalarType *>(&t) ||
               dynamic_cast<const FunctionType *>(&t)->returnTypeIDs.size() == 1);
        auto pRes = convert(t);
        // only allow static shape
        if (pRes.isa<TensorType>() && !pRes.dyn_cast<TensorType>().hasStaticShape())
        {
            return getElementTypeOrSelf(pRes);
        }
        return pRes;
    }

    ::mlir::Location MLIRGeneratorImpl::loc(Location loc)
    {
        return ::mlir::FileLineColLoc::get(
            builder.getStringAttr(loc.begin.filename == nullptr ? "" : *loc.begin.filename), loc.begin.line,
            loc.begin.column);
    }

    inline void MLIRGeneratorImpl::mlirGenBody(const std::vector<Statement> &block)
    {
        ScopedHashTableScope<StringRef, Value> var_scope(symbolTable);
        for (auto &stmt : block)
        {
            // TODO: Specific handling for variable declarations and return statement.

            visitor_gen(stmt);
        }
    }

    MLIRGeneratorImpl::result_variant MLIRGeneratorImpl::visitor_gen(const Expression &node)
    {
        auto visitor = MLIRGeneratorImpl(builder, module, symbolTable, funcTable, inferer);
        node.visit(visitor);
        if (std::holds_alternative<std::monostate>(visitor.result))
            throw MLIRGenerationException();
        return visitor.result;
    }

    MLIRGeneratorImpl::result_variant MLIRGeneratorImpl::visitor_gen(const Statement &node)
    {
        auto visitor = MLIRGeneratorImpl(builder, module, symbolTable, funcTable, inferer);
        node.visit(visitor);
        if (std::holds_alternative<std::monostate>(visitor.result))
            throw MLIRGenerationException();
        return visitor.result;
    }

    MLIRGeneratorImpl::result_variant MLIRGeneratorImpl::visitor_gen(const std::vector<Statement> &nodes)
    {
        ::mlir::SmallVector<Value> values;
        for (const auto &node : nodes)
        {
            auto visitor = MLIRGeneratorImpl(builder, module, symbolTable, funcTable, inferer);
            node.visit(visitor);
            if (std::holds_alternative<std::monostate>(visitor.result))
                throw MLIRGenerationException();
            values.push_back(std::get<Value>(visitor.result));
        }

        return values;
    }

    MLIRGeneratorImpl::result_variant MLIRGeneratorImpl::visitor_gen(const std::vector<Expression> &nodes)
    {
        ::mlir::SmallVector<Value> values;
        for (const auto &node : nodes)
        {
            auto visitor = MLIRGeneratorImpl(builder, module, symbolTable, funcTable, inferer);
            node.visit(visitor);
            if (std::holds_alternative<std::monostate>(visitor.result))
                throw MLIRGenerationException();
            values.push_back(std::get<Value>(visitor.result));
        }

        return values;
    }
    llvm::ArrayRef<int64_t> MLIRGeneratorImpl::getShape(const Value &lhs, const Value &rhs)
    {
        llvm::ArrayRef<int64_t> shape;
        if (lhs.getType().isa<::mlir::TensorType>() &&
            lhs.getType().dyn_cast<::mlir::RankedTensorType>().hasStaticShape())
        {
            shape = lhs.getType().dyn_cast<::mlir::RankedTensorType>().getShape();
        }
        else if (rhs.getType().isa<::mlir::TensorType>() &&
                 rhs.getType().dyn_cast<::mlir::RankedTensorType>().hasStaticShape())
        {
            shape = rhs.getType().dyn_cast<::mlir::RankedTensorType>().getShape();
        }
        else
        {
            shape = llvm::SmallVector<int64_t, 1>{-1};
        }
        return shape;
    }
    MLIRGeneratorImpl::MLIRGeneratorImpl(OpBuilder &builder,
                                         ModuleOp &module,
                                         llvm::ScopedHashTable<llvm::StringRef, ::mlir::Value> &symbolTable,
                                         llvm::StringMap<::mlir::func::FuncOp> &funcTable,
                                         const TypeInferer &inferer) :
        builder{builder}, module{module}, symbolTable{symbolTable}, funcTable{funcTable}, inferer{inferer}, result{}
    {
        (void) module;
    }

} // namespace voila::mlir
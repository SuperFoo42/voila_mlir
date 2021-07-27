#include "MLIRGeneratorImpl.hpp"

#include "MlirGenerationException.hpp"
#include "NotImplementedException.hpp"

#include <mlir/IR/BuiltinTypes.h>

namespace voila::mlir
{
    using namespace ast;
    using ::llvm::ScopedHashTableScope;
    using ::llvm::SmallVector;
    using ::llvm::StringRef;
    using ::mlir::Value;

    void MLIRGeneratorImpl::operator()(const AggrSum &sum)
    {
        auto location = loc(sum.get_location());

        mlir::Value expr = std::get<Value>(visitor_gen(sum.src));

        // TODO: cleaner solution
        ::mlir::Type resType;
        if (expr.getType().dyn_cast<::mlir::TensorType>().getElementType().isIntOrIndex())
        {
            resType = builder.getI64Type();
        }
        else if (expr.getType().dyn_cast<::mlir::TensorType>().getElementType().isIntOrFloat())
        {
            resType = builder.getF64Type();
        }
        else
        {
            throw MLIRGenerationException();
        }

        result = builder.create<::mlir::voila::SumOp>(location, resType, ::llvm::makeArrayRef(expr));
    }

    void MLIRGeneratorImpl::operator()(const AggrCnt &cnt)
    {
        auto location = loc(cnt.get_location());

        mlir::Value expr = std::get<Value>(visitor_gen(cnt.src));

        result = builder.create<::mlir::voila::SumOp>(location, builder.getI64Type(), ::llvm::makeArrayRef(expr));
    }

    void MLIRGeneratorImpl::operator()(const AggrMin &min)
    {
        auto location = loc(min.get_location());

        mlir::Value expr = std::get<Value>(visitor_gen(min.src));

        ::mlir::Type resType = expr.getType().dyn_cast<::mlir::TensorType>().getElementType();

        result = builder.create<::mlir::voila::SumOp>(location, resType, ::llvm::makeArrayRef(expr));
    }

    void MLIRGeneratorImpl::operator()(const AggrMax &max)
    {
        auto location = loc(max.get_location());

        mlir::Value expr = std::get<Value>(visitor_gen(max.src));
        ::mlir::Type resType = expr.getType().dyn_cast<::mlir::TensorType>().getElementType();

        result = builder.create<::mlir::voila::SumOp>(location, resType, ::llvm::makeArrayRef(expr));
    }

    void MLIRGeneratorImpl::operator()(const AggrAvg &avg)
    {
        auto location = loc(avg.get_location());

        mlir::Value expr = std::get<Value>(visitor_gen(avg.src));

        result = builder.create<::mlir::voila::AvgOp>(location, builder.getF64Type(), ::llvm::makeArrayRef(expr));
    }

    void MLIRGeneratorImpl::operator()(const Write &write)
    {
        ASTVisitor::operator()(write);
    }

    void MLIRGeneratorImpl::operator()(const Scatter &scatter)
    {
        ASTVisitor::operator()(scatter);
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
        result = builder.create<::mlir::voila::GenericCallOp>(location, call.fun, operands,
                                                              funcTable.at(call.fun).getType().getResult(0));
    }

    void MLIRGeneratorImpl::operator()(const Assign &assign)
    {
        auto res = visitor_gen(assign.expr);
        result = res;
        auto value = std::get<Value>(res);

        // assign value to variable
        if (assign.dest.is_variable())
        {
            symbolTable.insert(assign.dest.as_variable()->var, value);
        }
        else
        {
            builder.create<::mlir::voila::MoveOp>(
                value.getLoc(), value, symbolTable.lookup(assign.dest.as_reference()->ref.as_variable()->var));
        }
    }

    void MLIRGeneratorImpl::operator()(const Emit &emit)
    {
        auto location = loc(emit.get_location());

        // 'return' takes an optional expression, handle that case here.
        mlir::Value expr = std::get<Value>(visitor_gen(emit.expr));

        // Otherwise, this return operation has zero operands.
        builder.create<::mlir::voila::EmitOp>(location, ::llvm::makeArrayRef(expr));
        result = ::mlir::success();
    }

    void MLIRGeneratorImpl::operator()(const Loop &loop)
    {
        auto location = loc(loop.get_location());
        mlir::Value cond = std::get<Value>(visitor_gen(loop.pred));

        auto voilaLoop = builder.create<::mlir::voila::LoopOp>(location, builder.getI1Type(), cond);
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

        result =
            builder.create<::mlir::voila::AddOp>(location, getType(add), std::get<Value>(lhs), std::get<Value>(rhs));
    }

    void MLIRGeneratorImpl::operator()(const Sub &sub)
    {
        auto location = loc(sub.get_location());
        auto lhs = visitor_gen(sub.lhs);
        auto rhs = visitor_gen(sub.rhs);

        result =
            builder.create<::mlir::voila::SubOp>(location, getType(sub), std::get<Value>(lhs), std::get<Value>(rhs));
    }

    void MLIRGeneratorImpl::operator()(const Mul &mul)
    {
        auto location = loc(mul.get_location());
        auto lhs = visitor_gen(mul.lhs);
        auto rhs = visitor_gen(mul.rhs);

        result =
            builder.create<::mlir::voila::MulOp>(location, getType(mul), std::get<Value>(lhs), std::get<Value>(rhs));
    }

    void MLIRGeneratorImpl::operator()(const Div &div)
    {
        auto location = loc(div.get_location());
        auto lhs = visitor_gen(div.lhs);
        auto rhs = visitor_gen(div.rhs);

        result =
            builder.create<::mlir::voila::DivOp>(location, getType(div), std::get<Value>(lhs), std::get<Value>(rhs));
    }

    void MLIRGeneratorImpl::operator()(const Mod &mod)
    {
        auto location = loc(mod.get_location());
        auto lhs = visitor_gen(mod.lhs);
        auto rhs = visitor_gen(mod.rhs);

        result =
            builder.create<::mlir::voila::ModOp>(location, getType(mod), std::get<Value>(lhs), std::get<Value>(rhs));
    }

    void MLIRGeneratorImpl::operator()(const Eq &eq)
    {
        result = getCmpOp<::mlir::voila::EqOp>(eq);
    }

    void MLIRGeneratorImpl::operator()(const Neq &neq)
    {
        result = getCmpOp<::mlir::voila::NeqOp>(neq);
    }

    void MLIRGeneratorImpl::operator()(const Le &le)
    {
        result = getCmpOp<::mlir::voila::LeOp>(le);
    }

    void MLIRGeneratorImpl::operator()(const Ge &ge)
    {
        result = getCmpOp<::mlir::voila::GeOp>(ge);
    }

    void MLIRGeneratorImpl::operator()(const Leq &leq)
    {
        result = getCmpOp<::mlir::voila::LeqOp>(leq);
    }

    void MLIRGeneratorImpl::operator()(const Geq &geq)
    {
        result = getCmpOp<::mlir::voila::GeqOp>(geq);
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

            result = builder.create<::mlir::voila::AndOp>(
                location, ::mlir::RankedTensorType::get(shape, builder.getI1Type()), lhs, rhs);
        }
        else
            result = builder.create<::mlir::voila::AndOp>(location, builder.getI1Type(), lhs, rhs);
    }

    void MLIRGeneratorImpl::operator()(const Or &anOr)
    {
        auto location = loc(anOr.get_location());
        auto lhs = std::get<Value>(visitor_gen(anOr.lhs));
        auto rhs = std::get<Value>(visitor_gen(anOr.rhs));
        result = builder.create<::mlir::voila::OrOp>(location, ::mlir::RankedTensorType::get(-1, builder.getI1Type()),
                                                     lhs, rhs);
    }

    void MLIRGeneratorImpl::operator()(const Not &aNot)
    {
        auto location = loc(aNot.get_location());
        auto param = visitor_gen(aNot.param);

        result = builder.create<::mlir::voila::NotOp>(location, ::mlir::RankedTensorType::get(-1, builder.getI1Type()),
                                                      std::get<Value>(param));
    }

    void MLIRGeneratorImpl::operator()(const Hash &hash)
    {
        auto location = loc(hash.get_location());
        auto param = visitor_gen(hash.items);

        result = builder.create<::mlir::voila::HashOp>(
            location, ::mlir::RankedTensorType::get(-1, builder.getI64Type()), std::get<Value>(param));
    }

    void MLIRGeneratorImpl::operator()(const IntConst &intConst)
    {
        result =
            builder.create<::mlir::voila::IntConstOp>(loc(intConst.get_location()), builder.getI64Type(), intConst.val);
    }

    void MLIRGeneratorImpl::operator()(const BooleanConst &booleanConst)
    {
        result = builder.create<::mlir::voila::BoolConstOp>(loc(booleanConst.get_location()), builder.getI1Type(),
                                                            booleanConst.val);
    }

    void MLIRGeneratorImpl::operator()(const FltConst &fltConst)
    {
        result = builder.create<::mlir::voila::FltConstOp>(loc(fltConst.get_location()),
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

        result = builder.create<::mlir::voila::ReadOp>(location, col.getType(), col, idx);
    }

    void MLIRGeneratorImpl::operator()(const Gather &gather)
    {
        auto location = loc(gather.get_location());
        auto col = std::get<Value>(visitor_gen(gather.column));
        auto idx = std::get<Value>(visitor_gen(gather.idxs));
        result = builder.create<::mlir::voila::GatherOp>(location, col.getType(), col, idx);
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
        std::transform(
            fun.args.begin(), fun.args.end(),
            std::back_inserter(arg_types), [&](const auto &t) -> auto { return getType(*t.as_expr()); });
        auto func_type = builder.getFunctionType(arg_types, llvm::None);
        auto function = ::mlir::FuncOp::create(location, fun.name, func_type);
        funcTable.emplace(fun.name, function);
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

        ::mlir::voila::EmitOp emitOp;
        if (!entryBlock.empty())
            emitOp = dyn_cast<::mlir::voila::EmitOp>(entryBlock.back());
        if (!emitOp)
        {
            builder.create<::mlir::voila::EmitOp>(loc(fun.get_location()));
        }
        else if (emitOp.hasOperand())
        {
            // Otherwise, if this return operation has an operand then add a result to
            // the function.
            // TODO: get emit type
            function.setType(builder.getFunctionType(function.getType().getInputs(), convert(Type())));
        }

        result = function;
    }

    void MLIRGeneratorImpl::operator()(const Main &main)
    {
        // TODO: can we just slice main to fun, or do we have to consider some special properties of main?
        ASTVisitor::operator()(static_cast<Fun>(main));
    }

    void MLIRGeneratorImpl::operator()(const Selection &selection)
    {
        auto location = loc(selection.get_location());
        auto values = std::get<Value>(visitor_gen(selection.param));
        auto pred = std::get<Value>(visitor_gen(selection.pred));
        assert(pred.getType().isa<::mlir::TensorType>());

        result = builder.create<::mlir::voila::SelectOp>(
            location,
            ::mlir::RankedTensorType::get(-1, values.getType().dyn_cast<::mlir::TensorType>().getElementType()), values,
            pred);
    }

    void MLIRGeneratorImpl::operator()(const Lookup &lookup)
    {
        auto location = loc(lookup.get_location());
        auto table = visitor_gen(lookup.table);
        auto keys = visitor_gen(lookup.keys);

        result =
            builder.create<::mlir::voila::LookupOp>(location, ::mlir::RankedTensorType::get(-1, builder.getIndexType()),
                                                    std::get<Value>(table), std::get<Value>(keys));
    }

    void MLIRGeneratorImpl::operator()(const Insert &insert)
    {
        auto location = loc(insert.get_location());
        auto table = std::get<Value>(visitor_gen(insert.keys));

        result = builder.create<::mlir::voila::InsertOp>(
            location, ::mlir::MemRefType::get(-1, table.getType().dyn_cast<::mlir::TensorType>().getElementType()),
            table);
    }

    void MLIRGeneratorImpl::operator()(const Variable &variable)
    {
        // Register the value in the symbol table.
        // TODO: SSA
        declare(variable.var, nullptr);
        result = symbolTable.lookup(variable.var);
    }

    ::mlir::Type MLIRGeneratorImpl::getType(const ASTNode &node)
    {
        const auto astType = inferer.get_type(node);
        return convert(astType);
    }

    ::mlir::Type MLIRGeneratorImpl::convert(const Type &t)
    {
        switch (t.t)
        {
            case DataType::BOOL:
                if (t.ar.is_undef())
                {
                    return ::mlir::RankedTensorType::get(-1, builder.getI1Type());
                }
                else
                {
                    return ::mlir::RankedTensorType::get(t.ar.get_size(), builder.getI1Type());
                }
            case DataType::INT32:
                if (t.ar.is_undef())
                {
                    return ::mlir::RankedTensorType::get(-1, builder.getI32Type());
                }
                else
                {
                    return ::mlir::RankedTensorType::get(t.ar.get_size(), builder.getI32Type());
                }
            case DataType::INT64:
                if (t.ar.is_undef())
                {
                    return ::mlir::RankedTensorType::get(-1, builder.getI64Type());
                }
                else
                {
                    return ::mlir::RankedTensorType::get(t.ar.get_size(), builder.getI64Type());
                }
            case DataType::DBL:
                if (t.ar.is_undef())
                {
                    return ::mlir::RankedTensorType::get(-1, builder.getF64Type());
                }
                else
                {
                    return ::mlir::RankedTensorType::get(t.ar.get_size(), builder.getF64Type());
                }
            default:
                // TODO
                throw NotImplementedException();
        }
    }

    ::mlir::Location MLIRGeneratorImpl::loc(Location loc)
    {
        return ::mlir::FileLineColLoc::get(builder.getIdentifier(*loc.begin.filename), loc.begin.line,
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

} // namespace voila::mlir
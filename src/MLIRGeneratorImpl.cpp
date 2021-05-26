#include "MLIRGeneratorImpl.hpp"

#include "MlirGenerationException.hpp"
#include "NotImplementedException.hpp"

namespace voila::mlir
{
    using namespace ast;
    using ::llvm::ScopedHashTableScope;
    using ::llvm::SmallVector;
    using ::llvm::StringRef;
    using ::mlir::Value;

    void MLIRGeneratorImpl::operator()(const AggrSum &sum)
    {
        ASTVisitor::operator()(sum);
    }

    void MLIRGeneratorImpl::operator()(const AggrCnt &cnt)
    {
        ASTVisitor::operator()(cnt);
    }

    void MLIRGeneratorImpl::operator()(const AggrMin &min)
    {
        ASTVisitor::operator()(min);
    }

    void MLIRGeneratorImpl::operator()(const AggrMax &max)
    {
        ASTVisitor::operator()(max);
    }

    void MLIRGeneratorImpl::operator()(const AggrAvg &avg)
    {
        ASTVisitor::operator()(avg);
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
        SmallVector<Value, 4> operands;
        for (auto &expr : call.args)
        {
            auto arg = visitor_gen(expr);
            operands.push_back(std::get<Value>(arg));
        }

        result = builder.create<::mlir::voila::GenericCallOp>(location, call.fun, operands);
    }

    void MLIRGeneratorImpl::operator()(const Assign &assign)
    {
        // TODO
        auto var = visitor_gen(assign.dest);
        (void) var;

        mlir::Value value = std::get<Value>(visitor_gen(assign.expr));

        result = value;
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
        ASTVisitor::operator()(loop);
    }

    void MLIRGeneratorImpl::operator()(const StatementWrapper &wrapper)
    {
        result = visitor_gen(wrapper.expr);
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
            builder.create<::mlir::voila::AddOp>(location, getType(sub), std::get<Value>(lhs), std::get<Value>(rhs));
    }

    void MLIRGeneratorImpl::operator()(const Mul &mul)
    {
        auto location = loc(mul.get_location());
        auto lhs = visitor_gen(mul.lhs);
        auto rhs = visitor_gen(mul.rhs);

        result =
            builder.create<::mlir::voila::AddOp>(location, getType(mul), std::get<Value>(lhs), std::get<Value>(rhs));
    }

    void MLIRGeneratorImpl::operator()(const Div &div)
    {
        auto location = loc(div.get_location());
        auto lhs = visitor_gen(div.lhs);
        auto rhs = visitor_gen(div.rhs);

        result =
            builder.create<::mlir::voila::AddOp>(location, getType(div), std::get<Value>(lhs), std::get<Value>(rhs));
    }

    void MLIRGeneratorImpl::operator()(const Mod &mod)
    {
        auto location = loc(mod.get_location());
        auto lhs = visitor_gen(mod.lhs);
        auto rhs = visitor_gen(mod.rhs);

        result =
            builder.create<::mlir::voila::AddOp>(location, getType(mod), std::get<Value>(lhs), std::get<Value>(rhs));
    }

    void MLIRGeneratorImpl::operator()(const Eq &eq)
    {
        auto location = loc(eq.get_location());
        SmallVector<Value, 2> operands;
        SmallVector<::mlir::Type, 2> operandTypes;
        auto lhs = visitor_gen(eq.lhs);
        operands.push_back(std::get<Value>(lhs));
        operandTypes.push_back(std::get<Value>(lhs).getType());
        auto rhs = visitor_gen(eq.rhs);
        operands.push_back(std::get<Value>(rhs));
        operandTypes.push_back(std::get<Value>(rhs).getType());

        result = builder.create<::mlir::voila::EqOp>(location, operandTypes, operands);
    }

    void MLIRGeneratorImpl::operator()(const Neq &neq)
    {
        auto location = loc(neq.get_location());
        SmallVector<Value, 2> operands;
        SmallVector<::mlir::Type, 2> operandTypes;
        auto lhs = visitor_gen(neq.lhs);
        operands.push_back(std::get<Value>(lhs));
        operandTypes.push_back(std::get<Value>(lhs).getType());
        auto rhs = visitor_gen(neq.rhs);
        operands.push_back(std::get<Value>(rhs));
        operandTypes.push_back(std::get<Value>(rhs).getType());

        result = builder.create<::mlir::voila::NeqOp>(location, operandTypes, operands);
    }

    void MLIRGeneratorImpl::operator()(const Le &le)
    {
        auto location = loc(le.get_location());
        SmallVector<Value, 2> operands;
        SmallVector<::mlir::Type, 2> operandTypes;
        auto lhs = visitor_gen(le.lhs);
        operands.push_back(std::get<Value>(lhs));
        operandTypes.push_back(std::get<Value>(lhs).getType());
        auto rhs = visitor_gen(le.rhs);
        operands.push_back(std::get<Value>(rhs));
        operandTypes.push_back(std::get<Value>(rhs).getType());

        result = builder.create<::mlir::voila::LeOp>(location, operandTypes, operands);
    }

    void MLIRGeneratorImpl::operator()(const Ge &ge)
    {
        auto location = loc(ge.get_location());
        SmallVector<Value, 2> operands;
        SmallVector<::mlir::Type, 2> operandTypes;
        auto lhs = visitor_gen(ge.lhs);
        operands.push_back(std::get<Value>(lhs));
        operandTypes.push_back(std::get<Value>(lhs).getType());
        auto rhs = visitor_gen(ge.rhs);
        operands.push_back(std::get<Value>(rhs));
        operandTypes.push_back(std::get<Value>(rhs).getType());

        result = builder.create<::mlir::voila::GeOp>(location, operandTypes, operands);
    }

    void MLIRGeneratorImpl::operator()(const Leq &leq)
    {
        auto location = loc(leq.get_location());
        SmallVector<Value, 2> operands;
        SmallVector<::mlir::Type, 2> operandTypes;
        auto lhs = visitor_gen(leq.lhs);
        operands.push_back(std::get<Value>(lhs));
        operandTypes.push_back(std::get<Value>(lhs).getType());
        auto rhs = visitor_gen(leq.rhs);
        operands.push_back(std::get<Value>(rhs));
        operandTypes.push_back(std::get<Value>(rhs).getType());

        result = builder.create<::mlir::voila::LeqOp>(location, operandTypes, operands);
    }

    void MLIRGeneratorImpl::operator()(const Geq &geq)
    {
        auto location = loc(geq.get_location());
        SmallVector<Value, 2> operands;
        SmallVector<::mlir::Type, 2> operandTypes;
        auto lhs = visitor_gen(geq.lhs);
        operandTypes.push_back(std::get<Value>(lhs).getType());
        operands.push_back(std::get<Value>(lhs));
        auto rhs = visitor_gen(geq.rhs);
        operands.push_back(std::get<Value>(rhs));
        operandTypes.push_back(std::get<Value>(rhs).getType());

        result = builder.create<::mlir::voila::GeqOp>(location, operandTypes, operands);
    }

    void MLIRGeneratorImpl::operator()(const And &anAnd)
    {
        auto location = loc(anAnd.get_location());
        SmallVector<Value, 2> operands;
        auto lhs = visitor_gen(anAnd.lhs);
        operands.push_back(std::get<Value>(lhs));
        auto rhs = visitor_gen(anAnd.rhs);
        operands.push_back(std::get<Value>(rhs));

        result = builder.create<::mlir::voila::AndOp>(location, operands);
    }

    void MLIRGeneratorImpl::operator()(const Or &anOr)
    {
        auto location = loc(anOr.get_location());
        SmallVector<Value, 2> operands;
        auto lhs = visitor_gen(anOr.lhs);
        operands.push_back(std::get<Value>(lhs));
        auto rhs = visitor_gen(anOr.rhs);
        operands.push_back(std::get<Value>(rhs));

        result = builder.create<::mlir::voila::OrOp>(location, operands);
    }

    void MLIRGeneratorImpl::operator()(const Not &aNot)
    {
        auto location = loc(aNot.get_location());
        auto param = visitor_gen(aNot.param);

        result = builder.create<::mlir::voila::NotOp>(location, std::get<Value>(param));
    }

    void MLIRGeneratorImpl::operator()(const IntConst &aConst)
    {
        result = builder.create<::mlir::voila::IntConstOp>(loc(aConst.get_location()), ::mlir::RankedTensorType::get(-1,builder.getI64Type()), aConst.val);
    }

    void MLIRGeneratorImpl::operator()(const BooleanConst &aConst)
    {
        result = builder.create<::mlir::voila::BoolConstOp>(loc(aConst.get_location()),::mlir::RankedTensorType::get(-1,builder.getI1Type()), aConst.val);
    }

    void MLIRGeneratorImpl::operator()(const FltConst &aConst)
    {
        result = builder.create<::mlir::voila::FltConstOp>(loc(aConst.get_location()),::mlir::RankedTensorType::get(-1,builder.getF64Type()), builder.getF64FloatAttr(aConst.val));
    }

    void MLIRGeneratorImpl::operator()(const StrConst &)
    {
        throw NotImplementedException();
    }

    void MLIRGeneratorImpl::operator()(const Read &read)
    {
        ASTVisitor::operator()(read);
    }

    void MLIRGeneratorImpl::operator()(const Gather &gather)
    {
        ASTVisitor::operator()(gather);
    }

    void MLIRGeneratorImpl::operator()(const Ref &param)
    {
        if (auto variable = symbolTable.lookup(param.ref.as_variable()->var))
        {
            result = variable;
            return;
        }
        emitError(loc(param.get_location()), "error: unknown variable '") << param.ref.as_variable()->var << "'";
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
        llvm::SmallVector<::mlir::Type> arg_types(fun.args.size(), getType(fun));
        auto func_type = builder.getFunctionType(arg_types, llvm::None);
        auto function = ::mlir::FuncOp::create(location, fun.name, func_type);
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
        auto values = visitor_gen(selection.param);
        auto pred = visitor_gen(selection.pred);

        result = builder.create<::mlir::voila::SelectOp>(location, std::get<Value>(values).getType(),
                                                         std::get<Value>(values), std::get<Value>(pred));
    }

    void MLIRGeneratorImpl::operator()(const Variable &variable)
    {
        // TODO
        result = builder.create<::mlir::tensor::FromElementsOp>(loc(variable.get_location()), ::mlir::ValueRange());
        // Register the value in the symbol table.
        declare(variable.var, std::get<Value>(result));
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
                    return ::mlir::UnrankedTensorType::get(builder.getI1Type());
                }
                else
                {
                    return ::mlir::RankedTensorType::get(t.ar.get_size(), builder.getI1Type());
                }
            case DataType::INT32:
                if (t.ar.is_undef())
                {
                    return ::mlir::UnrankedTensorType::get(builder.getI32Type());
                }
                else
                {
                    return ::mlir::RankedTensorType::get(t.ar.get_size(), builder.getI32Type());
                }
            case DataType::INT64:
                if (t.ar.is_undef())
                {
                    return ::mlir::UnrankedTensorType::get(builder.getI64Type());
                }
                else
                {
                    return ::mlir::RankedTensorType::get(t.ar.get_size(), builder.getI64Type());
                }
            case DataType::DBL:
                if (t.ar.is_undef())
                {
                    return ::mlir::UnrankedTensorType::get(builder.getF64Type());
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
        auto visitor = MLIRGeneratorImpl(builder, module, symbolTable, inferer);
        node.visit(visitor);
        if (std::holds_alternative<std::monostate>(visitor.result))
            throw MLIRGenerationException();
        return visitor.result;
    }

    MLIRGeneratorImpl::result_variant MLIRGeneratorImpl::visitor_gen(const Statement &node)
    {
        auto visitor = MLIRGeneratorImpl(builder, module, symbolTable, inferer);
        node.visit(visitor);
        if (std::holds_alternative<std::monostate>(visitor.result))
            throw MLIRGenerationException();
        return visitor.result;
    }
} // namespace voila::mlir
#include "MLIRGeneratorImpl.hpp"

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
            if (std::holds_alternative<std::monostate>(arg))
                return;
            operands.push_back(std::get<Value>(arg));
        }

        result = builder.create<::mlir::voila::GenericCallOp>(location, call.fun, operands);
    }

    void MLIRGeneratorImpl::operator()(const Assign &assign) {
        //TODO
        auto var = visitor_gen(assign.dest);
        (void)var;

        mlir::Value value = std::get<Value>(visitor_gen(assign.expr));

        result = value;
    }

    void MLIRGeneratorImpl::operator()(const Emit &emit)
    {
        ASTVisitor::operator()(emit);
    }

    void MLIRGeneratorImpl::operator()(const Loop &loop)
    {
        ASTVisitor::operator()(loop);
    }

    void MLIRGeneratorImpl::operator()(const StatementWrapper &wrapper)
    {
        ASTVisitor::operator()(*wrapper.expr.as_expr());
    }

    void MLIRGeneratorImpl::operator()(const Add &add)
    {
        auto location = loc(add.get_location());
        SmallVector<Value, 2> operands;
        auto lhs = visitor_gen(add.lhs);
        if (std::holds_alternative<std::monostate>(lhs))
            return;
        operands.push_back(std::get<Value>(lhs));
        auto rhs = visitor_gen(add.rhs);
        if (std::holds_alternative<std::monostate>(rhs))
            return;
        operands.push_back(std::get<Value>(rhs));

        result = builder.create<::mlir::voila::AddOp>(location, operands);
    }

    void MLIRGeneratorImpl::operator()(const Sub &sub)
    {
        auto location = loc(sub.get_location());
        SmallVector<Value, 2> operands;
        auto lhs = visitor_gen(sub.lhs);
        if (std::holds_alternative<std::monostate>(lhs))
            return;
        operands.push_back(std::get<Value>(lhs));
        auto rhs = visitor_gen(sub.rhs);
        if (std::holds_alternative<std::monostate>(rhs))
            return;
        operands.push_back(std::get<Value>(rhs));

        result = builder.create<::mlir::voila::SubOp>(location, operands);
    }

    void MLIRGeneratorImpl::operator()(const Mul &mul)
    {
        auto location = loc(mul.get_location());
        SmallVector<Value, 2> operands;
        auto lhs = visitor_gen(mul.lhs);
        if (std::holds_alternative<std::monostate>(lhs))
            return;
        operands.push_back(std::get<Value>(lhs));
        auto rhs = visitor_gen(mul.rhs);
        if (std::holds_alternative<std::monostate>(rhs))
            return;
        operands.push_back(std::get<Value>(rhs));

        result = builder.create<::mlir::voila::MulOp>(location, operands);
    }

    void MLIRGeneratorImpl::operator()(const Div &div)
    {
        auto location = loc(div.get_location());
        SmallVector<Value, 2> operands;
        auto lhs = visitor_gen(div.lhs);
        if (std::holds_alternative<std::monostate>(lhs))
            return;
        operands.push_back(std::get<Value>(lhs));
        auto rhs = visitor_gen(div.rhs);
        if (std::holds_alternative<std::monostate>(rhs))
            return;
        operands.push_back(std::get<Value>(rhs));

        result = builder.create<::mlir::voila::DivOp>(location, operands);
    }

    void MLIRGeneratorImpl::operator()(const Mod &mod)
    {
        auto location = loc(mod.get_location());
        SmallVector<Value, 2> operands;
        auto lhs = visitor_gen(mod.lhs);
        if (std::holds_alternative<std::monostate>(lhs))
            return;
        operands.push_back(std::get<Value>(lhs));
        auto rhs = visitor_gen(mod.rhs);
        if (std::holds_alternative<std::monostate>(rhs))
            return;
        operands.push_back(std::get<Value>(rhs));

        result = builder.create<::mlir::voila::ModOp>(location, operands);
    }

    void MLIRGeneratorImpl::operator()(const Eq &eq)
    {
        auto location = loc(eq.get_location());
        SmallVector<Value, 2> operands;
        SmallVector<::mlir::Type, 2> operandTypes;
        auto lhs = visitor_gen(eq.lhs);
        if (std::holds_alternative<std::monostate>(lhs))
            return;

        operandTypes.push_back(std::get<Value>(lhs).getType());
        auto rhs = visitor_gen(eq.rhs);
        if (std::holds_alternative<std::monostate>(rhs))
            return;
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
        if (std::holds_alternative<std::monostate>(lhs))
            return;

        operandTypes.push_back(std::get<Value>(lhs).getType());
        auto rhs = visitor_gen(neq.rhs);
        if (std::holds_alternative<std::monostate>(rhs))
            return;
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
        if (std::holds_alternative<std::monostate>(lhs))
            return;

        operandTypes.push_back(std::get<Value>(lhs).getType());
        auto rhs = visitor_gen(le.rhs);
        if (std::holds_alternative<std::monostate>(rhs))
            return;
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
        if (std::holds_alternative<std::monostate>(lhs))
            return;

        operandTypes.push_back(std::get<Value>(lhs).getType());
        auto rhs = visitor_gen(ge.rhs);
        if (std::holds_alternative<std::monostate>(rhs))
            return;
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
        if (std::holds_alternative<std::monostate>(lhs))
            return;

        operandTypes.push_back(std::get<Value>(lhs).getType());
        auto rhs = visitor_gen(leq.rhs);
        if (std::holds_alternative<std::monostate>(rhs))
            return;
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
        if (std::holds_alternative<std::monostate>(lhs))
            return;

        operandTypes.push_back(std::get<Value>(lhs).getType());
        auto rhs = visitor_gen(geq.rhs);
        if (std::holds_alternative<std::monostate>(rhs))
            return;
        operands.push_back(std::get<Value>(rhs));
        operandTypes.push_back(std::get<Value>(rhs).getType());

        result = builder.create<::mlir::voila::GeqOp>(location, operandTypes, operands);
    }

    void MLIRGeneratorImpl::operator()(const And &anAnd)
    {
        auto location = loc(anAnd.get_location());
        SmallVector<Value, 2> operands;
        auto lhs = visitor_gen(anAnd.lhs);
        if (std::holds_alternative<std::monostate>(lhs))
            return;

        auto rhs = visitor_gen(anAnd.rhs);
        if (std::holds_alternative<std::monostate>(rhs))
            return;
        operands.push_back(std::get<Value>(rhs));

        result = builder.create<::mlir::voila::AndOp>(location, operands);
    }

    void MLIRGeneratorImpl::operator()(const Or &anOr)
    {
        auto location = loc(anOr.get_location());
        SmallVector<Value, 2> operands;
        auto lhs = visitor_gen(anOr.lhs);
        if (std::holds_alternative<std::monostate>(lhs))
            return;

        auto rhs = visitor_gen(anOr.rhs);
        if (std::holds_alternative<std::monostate>(rhs))
            return;
        operands.push_back(std::get<Value>(rhs));

        result = builder.create<::mlir::voila::OrOp>(location, operands);
    }

    void MLIRGeneratorImpl::operator()(const Not &aNot)
    {
        auto location = loc(aNot.get_location());
        auto param = visitor_gen(aNot.param);
        if (std::holds_alternative<std::monostate>(param))
            return;

        result = builder.create<::mlir::voila::NotOp>(location, std::get<Value>(param));
    }

    void MLIRGeneratorImpl::operator()(const IntConst &aConst)
    {
        result = builder.create<::mlir::voila::IntConstOp>(loc(aConst.get_location()), aConst.val);
    }

    void MLIRGeneratorImpl::operator()(const BooleanConst &aConst)
    {
        result = builder.create<::mlir::voila::BoolConstOp>(loc(aConst.get_location()), aConst.val);
    }

    void MLIRGeneratorImpl::operator()(const FltConst &aConst)
    {
        result = builder.create<::mlir::voila::FltConstOp>(loc(aConst.get_location()), aConst.val);
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
        if (!function)
        {
            return;
        }

        auto &entryBlock = *function.addEntryBlock();
        auto protoArgs = fun.args;

        // Declare all the function arguments in the symbol table.
        for (const auto nameValue : llvm::zip(protoArgs, entryBlock.getArguments()))
        {
            if (::mlir::failed(declare(std::get<0>(nameValue).as_variable()->var, std::get<1>(nameValue))))
                return;
        }

        builder.setInsertionPointToStart(&entryBlock);

        // Emit the body of the function.
        if (::mlir::failed(mlirGenBody(fun.body)))
        {
            function.erase();
            return;
        }

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
        //TODO: can we just slice main to fun, or do we have to consider some special properties of main?
        ASTVisitor::operator()(static_cast<Fun>(main));
    }

    void MLIRGeneratorImpl::operator()(const Selection &selection)
    {
        auto location = loc(selection.get_location());
        auto values = visitor_gen(selection.param);
        if (std::holds_alternative<std::monostate>(values))
            return;
        auto pred = visitor_gen(selection.pred);
        if (std::holds_alternative<std::monostate>(pred))
            return;

        result = builder.create<::mlir::voila::SelectOp>(location, std::get<Value>(values).getType(), std::get<Value>(values), std::get<Value>(pred));
    }

    void MLIRGeneratorImpl::operator()(const Variable &variable)
    {
        //TODO
        result = builder.create<::mlir::tensor::FromElementsOp>(loc(variable.get_location()), ::mlir::ValueRange());
        // Register the value in the symbol table.
        if (failed(declare(variable.var, std::get<Value>(result))))
            return;
    }

    ::mlir::Type MLIRGeneratorImpl::getType(const ASTNode &node)
    {
        const auto astType = inferer.get_type(node);
        return convert(astType);
    }

    ::mlir::Type MLIRGeneratorImpl::convert(Type t)
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

    inline ::mlir::LogicalResult MLIRGeneratorImpl::mlirGenBody(const std::vector<Statement> &block)
    {
        ScopedHashTableScope<StringRef, Value> var_scope(symbolTable);
        for (auto &stmt : block)
        {
            // Specific handling for variable declarations, return statement, and
            // print. These can only appear in block list and not in nested
            // expressions.
            if (stmt.is_assignment())
            {
                if (std::holds_alternative<std::monostate>(visitor_gen(stmt)))
                    return ::mlir::failure();
                continue;
            }

            if (stmt.is_emit())
            {
                return std::get<::mlir::LogicalResult>(visitor_gen(stmt));
            }

            if (std::holds_alternative<std::monostate>(visitor_gen(stmt)))
                return ::mlir::failure();
        }
        return ::mlir::success();
    }

    MLIRGeneratorImpl::result_variant MLIRGeneratorImpl::visitor_gen(const Expression &node)
    {
        auto visitor = MLIRGeneratorImpl(builder, module, symbolTable, inferer);
        node.visit(visitor);
        return visitor.result;
    }

    MLIRGeneratorImpl::result_variant MLIRGeneratorImpl::visitor_gen(const Statement &node)
    {
        auto visitor = MLIRGeneratorImpl(builder, module, symbolTable, inferer);
        node.visit(visitor);
        return visitor.result;
    }
} // namespace voila::mlir
#include "MLIRGeneratorImpl.hpp"
#include "ASTNodes.hpp"
#include "MlirGenerationException.hpp"
#include "NotImplementedException.hpp"
#include "TypeInferer.hpp"
#include "Types.hpp"
#include "ast/Add.hpp"
#include "ast/AggrAvg.hpp"
#include "ast/AggrCnt.hpp"
#include "ast/AggrMax.hpp"
#include "ast/AggrMin.hpp"
#include "ast/AggrSum.hpp"
#include "ast/And.hpp"
#include "ast/Assign.hpp"
#include "ast/BooleanConst.hpp"
#include "ast/Div.hpp"
#include "ast/Emit.hpp"
#include "ast/Eq.hpp"
#include "ast/FltConst.hpp"
#include "ast/Fun.hpp"
#include "ast/FunctionCall.hpp"
#include "ast/Gather.hpp"
#include "ast/Ge.hpp"
#include "ast/Geq.hpp"
#include "ast/Hash.hpp"
#include "ast/Insert.hpp"
#include "ast/IntConst.hpp"
#include "ast/Le.hpp"
#include "ast/Leq.hpp"
#include "ast/Lookup.hpp"
#include "ast/Loop.hpp"
#include "ast/Main.hpp"
#include "ast/Mod.hpp"
#include "ast/Mul.hpp"
#include "ast/Neq.hpp"
#include "ast/Not.hpp"
#include "ast/Or.hpp"
#include "ast/Predicate.hpp"
#include "ast/Read.hpp"
#include "ast/Ref.hpp"
#include "ast/Scatter.hpp"
#include "ast/Selection.hpp"
#include "ast/StatementWrapper.hpp"
#include "ast/Sub.hpp"
#include "ast/Variable.hpp"
#include "location.hpp"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialects/Voila/IR/VoilaOps.h"
#include "mlir/Dialects/Voila/Interfaces/PredicationOpInterface.hpp"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cassert>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/TypeUtilities.h>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

namespace mlir
{
    class ModuleOp;
}

namespace voila::ast
{
    class StrConst;
    class Write;
} // namespace voila::ast

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
    using std::to_string;

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<AggrSum> sum) { return createAggr(sum); }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<AggrCnt> cnt) { return createAggr(cnt); }

    // TODO: is this behaviour better than explicit type checking and throw on monostate?
    result_variant MLIRGeneratorImpl::visit_impl(std::monostate) { return Value(); }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<AggrMin> min) { return createAggr(min); }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<AggrMax> max) { return createAggr(max); }

    // TODO
    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<AggrAvg> avg) { return createAggr(avg); }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Write> ) { throw std::logic_error("Not implemented"); }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Scatter> scatter)
    {
        auto location = loc(scatter->get_location());
        auto col = std::get<Value>(std::visit(*this, scatter->src()));
        auto idx = std::get<Value>(std::visit(*this, scatter->idxs()));
        return builder.create<ScatterOp>(location, col.getType(), idx, col, Value());
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<FunctionCall> call)
    {
        auto location = loc(call->get_location());
        SmallVector<Value> operands;
        for (auto &expr : call->args())
        {
            auto arg = std::visit(*this, expr);
            operands.push_back(std::get<Value>(arg));
        }

        return builder
            .create<GenericCallOp>(location, call->fun(), operands,
                                   funcTable.lookup(call->fun()).getFunctionType().getResult(0))
            ->getResults();
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Assign> assign)
    {
        auto res = std::visit(*this, assign->expr());
        ValueRange val;
        if (std::holds_alternative<Value>(res))
            val = std::get<Value>(res);
        else if (std::holds_alternative<ValueRange>(res))
            val = std::get<ValueRange>(res);

        // assign value to variable
        for (const auto &en : llvm::enumerate(llvm::zip(assign->dests(), val)))
        {
            Value value;
            ASTNodeVariant dest;
            std::tie(dest, val) = en.value();

            symbolTable.insert(
                std::visit(overloaded{[](std::shared_ptr<Variable> &var) { return var; },
                                      [](std::shared_ptr<Ref> &ref)
                                      { return std::get<std::shared_ptr<Variable>>(ref->ref()); },
                                      [](auto &) -> std::shared_ptr<Variable> { throw std::bad_variant_access(); }},
                           dest)
                    ->var,
                value);
        }

        return res;
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Emit> emit)
    {
        auto location = loc(emit->get_location());

        // 'return' takes an optional expression, handle that case here.
        SmallVector<Value> exprs;
        for (auto &val : emit->exprs())
        {
            exprs.push_back(std::get<Value>(std::visit(*this, val)));
        }

        // Otherwise, this return operation has zero operands.
        builder.create<EmitOp>(location, exprs);
        return ::mlir::success();
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Loop> loop)
    {
        auto location = loc(loop->get_location());
        mlir::Value cond = std::get<Value>(std::visit(*this, loop->pred()));

        auto voilaLoop = builder.create<LoopOp>(location, builder.getI1Type(), cond);
        auto &bodyRegion = voilaLoop.getBody();
        bodyRegion.push_back(new ::mlir::Block());
        ::mlir::Block &bodyBlock = bodyRegion.front();
        ::mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&bodyBlock);
        for (const auto &elem : loop->stmts())
        {
            std::visit(*this, elem);
        }

        return ::mlir::success();
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<StatementWrapper> wrapper)
    {
        // forward to expr
        return std::visit(*this, wrapper->expr());
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Add> add)
    {
        auto location = loc(add->get_location());
        auto lhs = std::visit(*this, add->lhs());
        auto rhs = std::visit(*this, add->rhs());
        auto resType = getTypes(add);
        assert(resType.size() == 1);
        return builder.create<AddOp>(location, resType, std::get<Value>(lhs), std::get<Value>(rhs));
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Sub> sub)
    {
        auto location = loc(sub->get_location());
        auto lhs = std::visit(*this, sub->lhs());
        auto rhs = std::visit(*this, sub->rhs());
        auto resType = getTypes(sub);
        assert(resType.size() == 1);
        return builder.create<SubOp>(location, resType, std::get<Value>(lhs), std::get<Value>(rhs));
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Mul> mul)
    {
        auto location = loc(mul->get_location());
        auto lhs = std::visit(*this, mul->lhs());
        auto rhs = std::visit(*this, mul->rhs());
        auto resType = getTypes(mul);
        assert(resType.size() == 1);
        return builder.create<MulOp>(location, resType, std::get<Value>(lhs), std::get<Value>(rhs));
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Div> div)
    {
        auto location = loc(div->get_location());
        auto lhs = std::visit(*this, div->lhs());
        auto rhs = std::visit(*this, div->rhs());

        auto resType = getTypes(div);
        assert(resType.size() == 1);
        return builder.create<DivOp>(location, resType, std::get<Value>(lhs), std::get<Value>(rhs));
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Mod> mod)
    {
        auto location = loc(mod->get_location());
        auto lhs = std::visit(*this, mod->lhs());
        auto rhs = std::visit(*this, mod->rhs());
        auto resType = getTypes(mod);
        assert(resType.size() == 1);
        return builder.create<ModOp>(location, resType, std::get<Value>(lhs), std::get<Value>(rhs));
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Eq> eq) { return getCmpOp(*eq); }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Neq> neq) { return getCmpOp(*neq); }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Le> le) { return getCmpOp(*le); }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Ge> ge) { return getCmpOp(*ge); }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Leq> leq) { return getCmpOp(*leq); }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Geq> geq) { return getCmpOp(*geq); }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<And> anAnd)
    {
        auto location = loc(anAnd->get_location());
        auto lhs = std::get<::mlir::Value>(std::visit(*this, anAnd->lhs()));
        auto rhs = std::get<::mlir::Value>(std::visit(*this, anAnd->rhs()));
        if (lhs.getType().isa<::mlir::TensorType>() || rhs.getType().isa<::mlir::TensorType>())
        {
            ::mlir::ArrayRef<int64_t> shape;
            shape = getShape(lhs, rhs);

            return builder.create<AndOp>(location, ::mlir::RankedTensorType::get(shape, builder.getI1Type()), lhs, rhs);
        }
        else
            return builder.create<AndOp>(location, builder.getI1Type(), lhs, rhs);
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Or> anOr)
    {
        auto location = loc(anOr->get_location());
        auto lhs = std::get<Value>(std::visit(*this, anOr->lhs()));
        auto rhs = std::get<Value>(std::visit(*this, anOr->rhs()));

        return builder.create<OrOp>(location, ::mlir::RankedTensorType::get(ShapedType::kDynamic, builder.getI1Type()),
                                    lhs, rhs);
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Not> aNot)
    {
        auto location = loc(aNot->get_location());
        auto param = std::visit(*this, aNot->param());

        return builder.create<NotOp>(location, ::mlir::RankedTensorType::get(ShapedType::kDynamic, builder.getI1Type()),
                                     std::get<Value>(param));
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Hash> hash)
    {
        auto location = loc(hash->get_location());
        SmallVector<Value> params;
        for (auto &val : hash->items())
        {
            params.push_back(std::get<Value>(std::visit(*this, val)));
        }

        auto retType = ::mlir::RankedTensorType::get(params.front().getType().dyn_cast<ShapedType>().getShape(),
                                                     builder.getI64Type());
        return builder.create<HashOp>(location, retType, params);
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<IntConst> intConst)
    {
        return builder.create<IntConstOp>(loc(intConst->get_location()), getScalarType(intConst), intConst->val);
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<BooleanConst> booleanConst)
    {
        return builder.create<BoolConstOp>(loc(booleanConst->get_location()), builder.getI1Type(), booleanConst->val);
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<FltConst> fltConst)
    {
        return builder.create<FltConstOp>(loc(fltConst->get_location()), builder.getF64FloatAttr(fltConst->val));
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<StrConst>) { throw NotImplementedException(); }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Predicate> pred)
    {
        return std::get<Value>(std::visit(*this, pred->expr()));
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Read> read)
    {
        auto location = loc(read->get_location());
        auto col = std::get<Value>(std::visit(*this, read->column()));
        auto idx = std::get<Value>(std::visit(*this, read->idx()));

        return builder.create<ReadOp>(location, col.getType(), col, idx, Value());
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Gather> gather)
    {
        auto location = loc(gather->get_location());
        auto col = std::get<Value>(std::visit(*this, gather->column()));
        auto idx = std::get<Value>(std::visit(*this, gather->idxs()));
        return builder.create<GatherOp>(
            location, RankedTensorType::get(idx.getType().dyn_cast<TensorType>().getShape(), getElementTypeOrSelf(col)),
            col, idx, Value());
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Ref> param)
    {
        const auto var = std::get<std::shared_ptr<Variable>>(param->ref())->var;
        if (symbolTable.count(var))
        {
            auto variable = symbolTable.lookup(var);
            // FIXME
            return variable;
            return std::monostate(); // TODO
        }
        throw MLIRGenerationException();
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Fun> fun)
    {
        if (inferer.get_type(fun)->undef())
            return std::monostate(); // TODO

        ScopedHashTableScope<StringRef, Value> var_scope(symbolTable);

        auto fName = fun->name();
        auto location = loc(fun->loc);

        // generic function, the return type will be inferred later.
        // Arguments type are uniformly unranked tensors.

        llvm::SmallVector<::mlir::Type> arg_types;
        for (const auto &t : fun->args())
        {
            auto types = getTypes(t);
            arg_types.insert(arg_types.end(), types.begin(), types.end());
        }

        auto func_type = builder.getFunctionType(arg_types, std::nullopt);
        auto function = ::mlir::func::FuncOp::create(location, fName, func_type);
        funcTable.insert(std::make_pair(fName, function));
        assert(function);

        auto &entryBlock = *function.addEntryBlock();
        auto protoArgs = fun->args();

        // Declare all the function arguments in the symbol table.
        for (const auto nameValue : llvm::zip(protoArgs, entryBlock.getArguments()))
        {
            declare(std::get<std::shared_ptr<Variable>>(std::get<0>(nameValue))->var, std::get<1>(nameValue));
        }

        builder.setInsertionPointToStart(&entryBlock);

        // Emit the body of the function.
        mlirGenBody(fun->body());

        EmitOp emitOp;
        if (!entryBlock.empty())
            emitOp = dyn_cast<EmitOp>(entryBlock.back());
        if (!emitOp)
        {
            builder.create<EmitOp>(loc(fun->get_location()));
        }
        else if (emitOp.hasOperand())
        {
            // Otherwise, if this return operation has an operand then add a result to
            // the function.
            // TODO: get emit type
            function.setType(builder.getFunctionType(function.getFunctionType().getInputs(), getTypes(fun->result())));
        }

        return function;
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Main> main)
    {
        operator()(std::dynamic_pointer_cast<Fun>(main));
        return ::mlir::success();
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Selection> selection)
    {
        auto location = loc(selection->get_location());
        auto values = std::get<Value>(std::visit(*this, selection->param()));
        auto pred = std::get<Value>(std::visit(*this, selection->pred()));
        assert(pred.getType().isa<::mlir::TensorType>());

        return builder.create<::mlir::voila::SelectOp>(
            location, ::mlir::RankedTensorType::get(ShapedType::kDynamic, getElementTypeOrSelf(values)), values, pred);
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Lookup> lookup)
    {
        auto location = loc(lookup->get_location());
        SmallVector<Value> tables;
        for (auto &val : lookup->tables())
            tables.push_back(std::get<Value>(std::visit(*this, val)));

        auto hashes = std::get<Value>(std::visit(*this, lookup->hashes()));

        SmallVector<Value> values;
        for (auto &val : lookup->values())
            values.push_back(std::get<Value>(std::visit(*this, val)));

        return builder.create<LookupOp>(
            location,
            ::mlir::RankedTensorType::get(hashes.getType().dyn_cast<::mlir::TensorType>().getShape(),
                                          builder.getIndexType()),
            values, tables, hashes);
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Insert> insert)
    {
        auto location = loc(insert->get_location());
        auto table = std::get<Value>(std::visit(*this, insert->keys()));

        SmallVector<Value> data;
        for (auto &val : insert->values())
            data.push_back(std::get<Value>(std::visit(*this, val)));

        ::mlir::SmallVector<::mlir::Type> retTypes;
        for (auto val : data)
        {
            retTypes.push_back(::mlir::RankedTensorType::get(ShapedType::kDynamic, getElementTypeOrSelf(val)));
        }
        auto insertOp = builder.create<InsertOp>(location, retTypes, table, data);
        return insertOp.getHashtables();
    }

    result_variant MLIRGeneratorImpl::visit_impl(std::shared_ptr<Variable> variable)
    {
        // Register the value in the symbol table.
        // TODO: SSA
        declare(variable->var, nullptr);
        return symbolTable.lookup(variable->var);
    }

    std::vector<::mlir::Type> MLIRGeneratorImpl::getTypes(const ASTNodeVariant &node)
    {
        auto astType = inferer.get_type(node);
        std::vector<::mlir::Type> types;
        if (std::dynamic_pointer_cast<ScalarType>(astType))
        {
            types.push_back(convert(*astType));
        }
        else
        {
            for (auto tid : std::dynamic_pointer_cast<FunctionType>(astType)->returnTypeIDs)
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
                return ::mlir::RankedTensorType::get(ShapedType::kDynamic, builder.getI1Type());
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
                return ::mlir::RankedTensorType::get(ShapedType::kDynamic, builder.getI32Type());
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
                return ::mlir::RankedTensorType::get(ShapedType::kDynamic, builder.getI64Type());
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
                return ::mlir::RankedTensorType::get(ShapedType::kDynamic, builder.getF64Type());
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

    ::mlir::Type MLIRGeneratorImpl::getScalarType(const ASTNodeVariant &node)
    {
        const auto astType = inferer.get_type(node);
        return scalarConvert(astType);
    }

    ::mlir::Type MLIRGeneratorImpl::scalarConvert(const std::shared_ptr<::voila::Type> &t) { return scalarConvert(*t); }

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

    inline void MLIRGeneratorImpl::mlirGenBody(const std::vector<ASTNodeVariant> &block)
    {
        ScopedHashTableScope<StringRef, Value> var_scope(symbolTable);
        for (auto &stmt : block)
        {
            // TODO: Specific handling for variable declarations and return statement.

            std::visit(*this, stmt);
        }
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
            shape = ShapedType::kDynamic;
        }
        return shape;
    }

    MLIRGeneratorImpl::MLIRGeneratorImpl(OpBuilder &builder,
                                         ModuleOp &module,
                                         llvm::ScopedHashTable<llvm::StringRef, ::mlir::Value> &symbolTable,
                                         llvm::StringMap<::mlir::func::FuncOp> &funcTable,
                                         const TypeInferer &inferer)
        : builder{builder}, module{module}, symbolTable{symbolTable}, funcTable{funcTable}, inferer{inferer}
    {
        (void)module;
    }

} // namespace voila::mlir
#include "MLIRGenerator.hpp"

namespace voila
{
    using namespace ast;

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
        ASTVisitor::operator()(call);
    }
    void MLIRGeneratorImpl::operator()(const Assign &assign)
    {
        ASTVisitor::operator()(assign);
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
        ASTVisitor::operator()(wrapper);
    }
    void MLIRGeneratorImpl::operator()(const Add &add)
    {
        ASTVisitor::operator()(add);
    }
    void MLIRGeneratorImpl::operator()(const Sub &sub)
    {
        ASTVisitor::operator()(sub);
    }
    void MLIRGeneratorImpl::operator()(const Mul &mul)
    {
        ASTVisitor::operator()(mul);
    }
    void MLIRGeneratorImpl::operator()(const Div &div)
    {
        ASTVisitor::operator()(div);
    }
    void MLIRGeneratorImpl::operator()(const Mod &mod)
    {
        ASTVisitor::operator()(mod);
    }
    void MLIRGeneratorImpl::operator()(const Eq &eq)
    {
        ASTVisitor::operator()(eq);
    }
    void MLIRGeneratorImpl::operator()(const Neq &neq)
    {
        ASTVisitor::operator()(neq);
    }
    void MLIRGeneratorImpl::operator()(const Le &le)
    {
        ASTVisitor::operator()(le);
    }
    void MLIRGeneratorImpl::operator()(const Ge &ge)
    {
        ASTVisitor::operator()(ge);
    }
    void MLIRGeneratorImpl::operator()(const Leq &leq)
    {
        ASTVisitor::operator()(leq);
    }
    void MLIRGeneratorImpl::operator()(const Geq &geq)
    {
        ASTVisitor::operator()(geq);
    }
    void MLIRGeneratorImpl::operator()(const And &anAnd)
    {
        ASTVisitor::operator()(anAnd);
    }
    void MLIRGeneratorImpl::operator()(const Or &anOr)
    {
        ASTVisitor::operator()(anOr);
    }
    void MLIRGeneratorImpl::operator()(const Not &aNot)
    {
        ASTVisitor::operator()(aNot);
    }
    void MLIRGeneratorImpl::operator()(const IntConst &aConst)
    {
        ASTVisitor::operator()(aConst);
    }
    void MLIRGeneratorImpl::operator()(const BooleanConst &aConst)
    {
        ASTVisitor::operator()(aConst);
    }
    void MLIRGeneratorImpl::operator()(const FltConst &aConst)
    {
        ASTVisitor::operator()(aConst);
    }
    void MLIRGeneratorImpl::operator()(const StrConst &aConst)
    {
        ASTVisitor::operator()(aConst);
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
        ASTVisitor::operator()(param);
    }
    void MLIRGeneratorImpl::operator()(const TupleGet &get)
    {
        ASTVisitor::operator()(get);
    }
    void MLIRGeneratorImpl::operator()(const TupleCreate &create)
    {
        ASTVisitor::operator()(create);
    }
    void MLIRGeneratorImpl::operator()(const Fun &fun)
    {
        ::llvm::ScopedHashTableScope<::llvm::StringRef, ::mlir::Value> var_scope(symbolTable);

        auto location = loc(fun.loc);

        // generic function, the return type will be inferred later.
        // Arguments type are uniformly unranked tensors.
        llvm::SmallVector<mlir::Type, 4> arg_types(fun.args.size(), getType(fun));
        auto func_type = builder.getFunctionType(arg_types, llvm::None);
        auto function = mlir::FuncOp::create(location, fun.name, func_type);
        if (!function)
        {
            return;
        }

        auto &entryBlock = *function.addEntryBlock();
        auto protoArgs = fun.args;

        // Declare all the function arguments in the symbol table.
        for (const auto nameValue : llvm::zip(protoArgs, entryBlock.getArguments()))
        {
            if (mlir::failed(declare(std::get<0>(nameValue).as_variable()->var, std::get<1>(nameValue))))
                return;
        }
    }

    void MLIRGeneratorImpl::operator()(const Main &main)
    {
        ASTVisitor::operator()(main);
    }
    void MLIRGeneratorImpl::operator()(const Selection &selection)
    {
        ASTVisitor::operator()(selection);
    }
    void MLIRGeneratorImpl::operator()(const Variable &variable)
    {
        ASTVisitor::operator()(variable);
    }
    mlir::OwningModuleRef MLIRGenerator::generate(const Program &program)
    {
        // We create an empty MLIR module and codegen functions one at a time and
        // add them to the module.
        module = mlir::ModuleOp::create(builder.getUnknownLoc());

        for (const auto &f : program.functions)
        {
            auto generatorImpl = MLIRGeneratorImpl(builder, module, symbolTable, program.inferer);
            generatorImpl(*f);
            auto genRes = generatorImpl.getValue();
            // TODO: error handling
            if (genRes.valueless_by_exception())
                return nullptr;
            assert(holds_alternative<::mlir::FuncOp>(genRes));
            module.push_back(std::get<::mlir::FuncOp>(genRes));
        }

        // Verify the module after we have finished constructing it, this will check
        // the structural properties of the IR and invoke any specific verifiers we
        // have on the Toy operations.
        if (failed(mlir::verify(module)))
        {
            module.emitError("module verification error");
            return nullptr;
        }

        return module;
    }
} // namespace voila

#include "MLIRGenerator.hpp"
#include "MLIRGeneratorImpl.hpp"
namespace voila
{
    using mlir::MLIRGeneratorImpl;
    ::mlir::OwningModuleRef MLIRGenerator::generate(const Program &program)
    {
        // We create an empty MLIR module and codegen functions one at a time and
        // add them to the module.
        module = ::mlir::ModuleOp::create(builder.getUnknownLoc());

        for (const auto &f : program.functions)
        {
            auto generatorImpl = MLIRGeneratorImpl(builder, module, symbolTable, program.inferer);
            generatorImpl(*f);
            auto genRes = generatorImpl.getValue();
            // TODO: error handling
            if (std::holds_alternative<std::monostate>(genRes))
                return nullptr;
            assert(std::holds_alternative<::mlir::FuncOp>(genRes));
            module.push_back(std::get<::mlir::FuncOp>(genRes));
        }

        // Verify the module after we have finished constructing it, this will check
        // the structural properties of the IR and invoke any specific verifiers we
        // have on the Toy operations.
        if (failed(::mlir::verify(module)))
        {
            module.emitError("module verification error");
            return nullptr;
        }

        return module;
    }
    ::mlir::OwningModuleRef MLIRGenerator::mlirGen(::mlir::MLIRContext &ctx, const Program &program)
    {
        MLIRGenerator generator(ctx);
        return generator.generate(program);
    }
    MLIRGenerator::MLIRGenerator(::mlir::MLIRContext &ctx) : builder{&ctx}
    {
        (void) this->builder;
    }
} // namespace voila

#include "MLIRGenerator.hpp"

#include "MLIRGeneratorImpl.hpp"
#include "MlirGenerationException.hpp"
#include "MlirModuleVerificationError.hpp"

#include <mlir/IR/Verifier.h>
namespace voila
{
    using namespace ast;
    using mlir::MLIRGeneratorImpl;
    ::mlir::OwningModuleRef MLIRGenerator::generate(const Program &program)
    {
        // We create an empty MLIR module and codegen functions one at a time and
        // add them to the module.
        module = ::mlir::ModuleOp::create(builder.getUnknownLoc());

        for (const auto &f : program.get_funcs())
        {
            auto generatorImpl = MLIRGeneratorImpl(builder, module, symbolTable, funcTable, program.inferer);
            generatorImpl(*f);
            auto genRes = generatorImpl.getValue();
            // TODO: error handling
            if (std::holds_alternative<std::monostate>(genRes))
                throw MLIRGenerationException();
            assert(std::holds_alternative<::mlir::FuncOp>(genRes));
            module.push_back(std::get<::mlir::FuncOp>(genRes));
        }

        // Verify the module after we have finished constructing it, this will check
        // the structural properties of the IR and invoke any specific verifiers we
        // have on the Voila operations.
        if (failed(::mlir::verify(module)))
        {
            throw MLIRModuleVerificationError();
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

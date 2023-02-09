#include "MLIRGenerator.hpp"
#include "MLIRGeneratorImpl.hpp"                // for MLIRGeneratorImpl
#include "MlirModuleVerificationError.hpp"      // for MLIRModuleVerificati...
#include "Program.hpp"                          // for Program
#include "VariableAlreadyDeclaredException.hpp" // for ast
#include "mlir/Support/LogicalResult.h"         // for failed, LogicalResult
#include "range/v3/iterator/basic_iterator.hpp" // for operator!=, basic_it...
#include "range/v3/range_fwd.hpp"               // for values_view
#include "range/v3/view/adaptor.hpp"            // for adaptor_cursor
#include "range/v3/view/facade.hpp"             // for facade_iterator_t
#include <cassert>                              // for assert
#include <memory>                               // for __shared_ptr_access
#include <mlir/IR/Verifier.h>                   // for verify
#include <unordered_map>                        // for operator==
#include <variant>                              // for monostate, holds_alt...

namespace mlir
{
    class MLIRContext;
} // namespace mlir

namespace voila
{
    using namespace ast;

    using mlir::MLIRGeneratorImpl;
    using ::mlir::ModuleOp;
    using ::mlir::OwningOpRef;
    OwningOpRef<ModuleOp> MLIRGenerator::generate(const Program &program)
    {
        // We create an empty MLIR module and codegen functions one at a time and
        // add them to the module.
        module = ::mlir::ModuleOp::create(builder.getUnknownLoc());

        for (const auto &f : program.life_funcs())
        {
            auto generatorImpl = MLIRGeneratorImpl(builder, module, symbolTable, funcTable, program.inferer);
            generatorImpl(*f);
            auto genRes = generatorImpl.getValue();
            // TODO: error handling
            if (std::holds_alternative<std::monostate>(genRes))
                continue; // unspecified function, ignore
            assert(std::holds_alternative<::mlir::func::FuncOp>(genRes));
            module.push_back(std::get<::mlir::func::FuncOp>(genRes));
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

    OwningOpRef<ModuleOp> MLIRGenerator::mlirGen(::mlir::MLIRContext &ctx, const Program &program)
    {
        MLIRGenerator generator(ctx);
        return generator.generate(program);
    }

    MLIRGenerator::MLIRGenerator(::mlir::MLIRContext &ctx) : builder{&ctx} { (void)this->builder; }
} // namespace voila

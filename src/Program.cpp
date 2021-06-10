#include "Program.hpp"

#include "MainFunctionNotFoundException.hpp"

#include <fstream>
#include <llvm/IR/AssemblyAnnotationWriter.h>
#include <spdlog/spdlog.h>

namespace voila
{
    using namespace ast;
    using namespace ::mlir;

    static std::string MLIRModuleToString(OwningModuleRef &module)
    {
        std::error_code ec;
        std::string res = "";
        llvm::raw_string_ostream os(res);
        module->print(os);
        os.flush();
        return res;
    }

    static std::string LLVMModuleToString(llvm::Module &module)
    {
        std::error_code ec;
        std::string res = "";
        llvm::raw_string_ostream os(res);
        llvm::AssemblyAnnotationWriter aw;
        module.print(os, &aw);
        os.flush();
        return res;
    }

    void Program::infer_type(const Expression &node)
    {
        node.visit(inferer);
    }

    void Program::infer_type(const ASTNode &node)
    {
        node.visit(inferer);
    }

    void Program::infer_type(const Statement &node)
    {
        node.visit(inferer);
    }

    void Program::to_dot(const std::string &fname)
    {
        for (auto &func : functions)
        {
            DotVisualizer vis(*func, std::optional<std::reference_wrapper<TypeInferer>>(inferer));
            std::ofstream out(fname + "." + func->name + ".dot", std::ios::out);
            out << vis;
            out.close();
        }
    }

    bool Program::has_var(const std::string &var_name)
    {
        return func_vars.contains(var_name);
    }

    Expression Program::get_var(const std::string &var_name)
    {
        return func_vars.at(var_name);
    }

    void Program::add_var(Expression expr)
    {
        assert(expr.is_variable());
        func_vars.emplace(expr.as_variable()->var, expr);
    }

    void Program::set_main_args_shape(const std::unordered_map<std::string, size_t> &shapes)
    {
        const auto main = std::find_if(
            functions.begin(), functions.end(), [](const auto &f) -> auto { return dynamic_cast<Main *>(f.get()); });
        if (main == functions.end())
            throw MainFunctionNotFoundException();
        auto &args = (*main)->args;
        for (auto &arg : args)
        {
            assert(arg.is_variable());
            if (shapes.contains(arg.as_variable()->var))
            {
                inferer.set_arity(arg.as_expr(), shapes.at(arg.as_variable()->var));
            }
        }
    }

    void Program::set_main_args_type(const std::unordered_map<std::string, DataType> &types)
    {
        const auto main = std::find_if(
            functions.begin(), functions.end(), [](const auto &f) -> auto { return dynamic_cast<Main *>(f.get()); });
        if (main == functions.end())
            throw MainFunctionNotFoundException();
        auto &args = (*main)->args;
        for (auto &arg : args)
        {
            assert(arg.is_variable());
            if (types.contains(arg.as_variable()->var))
            {
                inferer.set_type(arg.as_expr(), types.at(arg.as_variable()->var));
            }
        }
    }
    const OwningModuleRef &Program::getMLIRModule() const
    {
        return mlirModule;
    }
    const MLIRContext &Program::getMLIRContext() const
    {
        return context;
    }
    void Program::runJIT(void *args, bool optimize)
    {
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();

        // Register the translation from MLIR to LLVM IR, which must happen before we
        // can JIT-compile.
        registerLLVMDialectTranslation(*mlirModule->getContext());

        // An optimization pipeline to use within the execution engine.
        auto optPipeline = makeOptimizingTransformer(
            /*optLevel=*/optimize ? 3 : 0, /*sizeLevel=*/0,
            /*targetMachine=*/nullptr);

        // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
        // the module.
        auto maybeEngine = ExecutionEngine::create(*mlirModule, /*llvmModuleBuilder=*/nullptr, optPipeline);
        assert(maybeEngine && "failed to construct an execution engine");
        auto &engine = maybeEngine.get();

        // Invoke the JIT-compiled function.
        auto invocationResult = engine->invokePacked("main", args);
        if (invocationResult)
        {
            throw JITInvocationError();
        }
    }
    void Program::convertToLLVM(bool optimize)
    {
        // lower to llvm
        registerLLVMDialectTranslation(*mlirModule->getContext());

        // Convert the module to LLVM IR in a new LLVM IR context.
        llvmModule = translateModuleToLLVMIR(*mlirModule, llvmContext);
        if (!llvmModule)
        {
            throw LLVMGenerationError();
        }

        // Initialize LLVM targets.
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();
        ExecutionEngine::setupTargetTriple(llvmModule.get());

        /// Optionally run an optimization pipeline over the llvm module.
        auto optPipeline = makeOptimizingTransformer(
            /*optLevel=*/optimize ? 3 : 0, /*sizeLevel=*/0,
            /*targetMachine=*/nullptr);
        if (auto err = optPipeline(llvmModule.get()))
        {
            throw err; //TODO: rethrow with other exception
        }
        //TODO:
        spdlog::debug(LLVMModuleToString(*llvmModule));
    }
    void Program::printLLVM(const std::string &filename)
    {
        std::error_code ec;
        llvm::raw_fd_ostream os(filename + ".llvm", ec, llvm::sys::fs::OF_None);
        llvm::AssemblyAnnotationWriter aw;
        llvmModule->print(os, &aw);
        os.flush();
    }
    void Program::printMLIR(const std::string &filename)
    {
        std::error_code ec;
        llvm::raw_fd_ostream os(filename + ".mlir", ec, llvm::sys::fs::OF_None);
        mlirModule->print(os);
        os.flush();
    }

    void Program::lowerMLIR(bool optimize)
    {
        ::mlir::PassManager pm(&context);
        // Apply any generic pass manager command line options and run the pipeline.
        applyPassManagerCLOptions(pm);
        pm.addPass(createInlinerPass());
        ::mlir::OpPassManager &optPM = pm.nest<FuncOp>();
        // Now that there is only one function, we can infer the shapes of each of
        // the operations.
        optPM.addPass(::voila::mlir::createShapeInferencePass()); // TODO: more inference?
        optPM.addPass(createCanonicalizerPass());
        optPM.addPass(createCSEPass());

        // Partially lower voila to affine with a few cleanups
        optPM.addPass(::voila::mlir::createLowerToAffinePass());
        optPM.addPass(createCanonicalizerPass());
        optPM.addPass(createCSEPass());

        // bufferization passes
        pm.addPass(createTensorConstantBufferizePass());
        optPM.addPass(createTensorBufferizePass());
        optPM.addPass(createStdBufferizePass());
        pm.addPass(createFuncBufferizePass());

        // optPM.addPass(mlir::createBufferDeallocationPass());

        auto state = pm.run(*mlirModule);
        spdlog::debug(MLIRModuleToString(mlirModule));

        if (failed(state))
        {
            throw MLIRLoweringError();
        }

        // FIXME: properly apply passes
        ::mlir::PassManager secondpm(&context);
        // Apply any generic pass manager command line options and run the pipeline.
        applyPassManagerCLOptions(secondpm);
        ::mlir::OpPassManager &secondOptPM = secondpm.nest<FuncOp>();
        // optPM.addPass(mlir::createCanonicalizerPass());
        // optPM.addPass(mlir::createCSEPass());

        // optPM.addPass(mlir::createAffineLoopNormalizePass());
        // optPM.addPass(mlir::createAffineLoopInvariantCodeMotionPass());
        if (optimize)
        {
            secondOptPM.addPass(createBufferHoistingPass());
            secondOptPM.addPass(createMemRefDataFlowOptPass());
            secondOptPM.addPass(createLoopFusionPass());
            secondOptPM.addPass(createLoopCoalescingPass());
            secondOptPM.addPass(createAffineParallelizePass());
        }
        secondOptPM.addPass(createFinalizingBufferizePass());
        secondpm.addPass(::voila::mlir::createLowerToLLVMPass());

        state = secondpm.run(*mlirModule);
        spdlog::debug(MLIRModuleToString(mlirModule));
        if (failed(state))
        {
            throw MLIRLoweringError();
        }
    }

    ::mlir::OwningModuleRef &Program::generateMLIR()
    {
        // Load our Dialect in this MLIR Context.
        context.getOrLoadDialect<::mlir::voila::VoilaDialect>();
        mlirModule = ::voila::MLIRGenerator::mlirGen(context, *this);

        if (!mlirModule)
            throw ::voila::MLIRGenerationError();
        spdlog::debug(MLIRModuleToString(mlirModule));
        return mlirModule;
    }
} // namespace voila
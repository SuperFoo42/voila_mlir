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
        std::string res;
        llvm::raw_string_ostream os(res);
        module->print(os);
        os.flush();
        return res;
    }

    static std::string LLVMModuleToString(llvm::Module &module)
    {
        std::string res;
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
    void Program::runJIT(bool optimize, [[maybe_unused]] std::optional<std::string> objPath)
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
        auto maybeEngine = ExecutionEngine::create(*mlirModule, /*llvmModuleBuilder=*/nullptr, optPipeline, llvm::None,
                                                   {}, true, true);
        assert(maybeEngine && "failed to construct an execution engine");
        auto engine = std::move(*maybeEngine);
        // FIXME: detor of fn results in segfault
        /*
                auto fn = engine->lookup("main");
                if (!fn)
                {
                    auto err = fn.takeError();
                    throw JITInvocationError();
                };
        */

        /*if (objPath.has_value())
            engine->dumpToObjectFile(objPath.value() + std::string(".main.o"));*/
        // Invoke the JIT-compiled function.
        // can not use new, because it appears that new does not really allocate storage in this case
//WTF: we need to pass all this to function in order to obtain a result
        auto *arg = static_cast<uint64_t *>(std::malloc(sizeof(uint64_t) * 100));
        std::fill_n(arg, 100, 123);
        auto *arg2 = static_cast<uint64_t *>(std::malloc(sizeof(uint64_t) * 100));
        std::fill_n(arg2, 100, 123);
        uint64_t foo = 0;
        uint64_t bar = 0;
        uint64_t baz = 0;
        SmallVector<void *> args;
        struct
        {
            uint64_t *p1;
            uint64_t *p2;
            uint64_t x; //?
            uint64_t p1_sizes[1]; //?
            uint64_t p2_sizes[1]; //?

        } res{};
        // pass pointers to args
        args.push_back(&arg);
        args.push_back(&arg2);
        args.push_back(&foo);
        args.push_back(&bar);
        args.push_back(&baz);
        args.push_back(&res);

        auto invocationResult = engine->invokePacked("main", args);

        if (invocationResult)
        {
            throw JITInvocationError();
        }

        for (auto i = 0; i < 100; ++i)
        {
            std::cout << arg2[i] << std::endl;
            std::cout << res.p1[i] << std::endl;
        }
        std::free(arg);
        std::free(arg2);
        std::free(res.p1);
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
            throw err; // TODO: rethrow with other exception
        }
        // TODO:
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

    void Program::lowerMLIR([[maybe_unused]] bool optimize)
    {
        ::mlir::PassManager pm(&context);
        if (debug)
        {
            pm.enableStatistics();
            // pm.enableIRPrinting();
        }
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
        optPM.addPass(createSCFBufferizePass());
        pm.addPass(createFuncBufferizePass());
        pm.addPass(createNormalizeMemRefsPass());
        optPM.addPass(createPipelineDataTransferPass());
        optPM.addPass(createCanonicalizerPass());
        optPM.addPass(createCSEPass());
        auto state = pm.run(*mlirModule);
        spdlog::debug(MLIRModuleToString(mlirModule));

        if (failed(state))
        {
            throw MLIRLoweringError();
        }

        // FIXME: properly apply passes
        ::mlir::PassManager secondpm(&context);
        if (debug)
        {
            secondpm.enableStatistics();
            // secondpm.enableIRPrinting();
        }
        // Apply any generic pass manager command line options and run the pipeline.
        applyPassManagerCLOptions(secondpm);
        ::mlir::OpPassManager &secondOptPM = secondpm.nest<FuncOp>();
        secondOptPM.addPass(createFinalizingBufferizePass());
        secondOptPM.addPass(createCanonicalizerPass());
        secondOptPM.addPass(createCSEPass());
        if (optimize)
        {
            secondOptPM.addPass(createPromoteBuffersToStackPass());
            secondpm.addPass(createBufferResultsToOutParamsPass());
            secondOptPM.addPass(createBufferHoistingPass());
            secondOptPM.addPass(createAffineDataCopyGenerationPass());
            secondOptPM.addPass(createAffineLoopInvariantCodeMotionPass());
            secondOptPM.addPass(createAffineLoopNormalizePass());
            secondOptPM.addPass(createLoopFusionPass());
            secondOptPM.addPass(createLoopCoalescingPass());
            secondOptPM.addPass(createLoopUnrollPass());
            secondOptPM.addPass(createLoopUnrollAndJamPass());
            // secondOptPM.addPass(createAffineParallelizePass());
            // secondOptPM.addPass(createSuperVectorizePass());
            secondOptPM.addPass(createSimplifyAffineStructuresPass());
            secondOptPM.addPass(createForLoopSpecializationPass());
            /*secondOptPM.addPass(createParallelLoopFusionPass());
            secondOptPM.addPass(createParallelLoopSpecializationPass());
            secondOptPM.addPass(createParallelLoopTilingPass());*/
            secondOptPM.addPass(createCanonicalizerPass());
            secondOptPM.addPass(createCSEPass());
        }
        secondOptPM.addPass(createBufferDeallocationPass());
        secondOptPM.addPass(createLowerAffinePass());
        secondOptPM.addPass(createLowerToCFGPass());

        secondpm.addPass(::voila::mlir::createLowerToLLVMPass());
        secondOptPM.addPass(createCanonicalizerPass());
        secondOptPM.addPass(createCSEPass());

        state = secondpm.run(*mlirModule);
        spdlog::warn(MLIRModuleToString(mlirModule));
        if (failed(state))
        {
            throw MLIRLoweringError();
        }
    }

    ::mlir::OwningModuleRef &Program::generateMLIR()
    {
        // Load our Dialect in this MLIR Context.
        context.getOrLoadDialect<::mlir::voila::VoilaDialect>();
        if (debug)
        {
            // context.disableMultithreading(); //FIXME: with threading disabled, the program segfaults
        }
        mlirModule = ::voila::MLIRGenerator::mlirGen(context, *this);

        if (!mlirModule)
            throw ::voila::MLIRGenerationError();
        spdlog::debug(MLIRModuleToString(mlirModule));
        return mlirModule;
    }
    Program &Program::operator<<(const Parameter param)
    {
        const auto main = std::find_if(
            functions.begin(), functions.end(), [](const auto &f) -> auto { return dynamic_cast<Main *>(f.get()); });
        if (main == functions.end())
            throw MainFunctionNotFoundException();
        auto &args = (*main)->args;
        auto arg = args.at(params.size());

        assert(arg.is_variable());
        params.push_back(param.data);
        inferer.set_type(arg.as_expr(), param.type);
        inferer.set_arity(arg.as_expr(), param.size);

        return *this;
    }

    std::unique_ptr<void *> Program::operator()()
    {
        runJIT(m_optimize, std::nullopt);
        // TODO:
        return std::make_unique<void *>(nullptr);
    }

    void Program::add_func(ast::Fun *f)
    {
        functions.emplace_back(f);
        f->variables = std::move(func_vars);
        func_vars.clear();
    }

    Parameter make_param(void *data, size_t size, DataType type)
    {
        return Parameter(data, size, type);
    }
} // namespace voila
#include "Program.hpp"

#include "Config.hpp"
#include "voila_lexer.hpp"

#include <MlirModuleVerificationError.hpp>
#include <utility>

namespace voila
{
    using namespace ast;
    using namespace ::mlir;
    using namespace ::voila::lexer;

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
        for (auto &func : get_funcs())
        {
            DotVisualizer vis(*func, std::optional<std::reference_wrapper<TypeInferer>>(inferer));
            std::ofstream out(fname + "." + func->name + ".dot", std::ios::out);
            out << vis;
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

    const OwningModuleRef &Program::getMLIRModule() const
    {
        return mlirModule;
    }

    const MLIRContext &Program::getMLIRContext() const
    {
        return context;
    }

    std::unique_ptr<ExecutionEngine> &Program::getOrCreateExecutionEngine()
    {
        if (!maybeEngine)
        {
            llvm::InitializeNativeTarget();
            llvm::InitializeNativeTargetAsmPrinter();

            // Register the translation from MLIR to LLVM IR, which must happen before we
            // can JIT-compile.
            registerLLVMDialectTranslation(*mlirModule->getContext());

            // An optimization pipeline to use within the execution engine.
            auto optPipeline = makeOptimizingTransformer(
                /*optLevel=*/config.optimize ? 3 : 0, /*sizeLevel=*/0,
                /*targetMachine=*/nullptr);

            // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
            // the module.
            auto expectedEngine = ExecutionEngine::create(*mlirModule, /*llvmModuleBuilder=*/nullptr, optPipeline,
                                                          llvm::None, {}, true, false);
            assert(expectedEngine && "failed to construct an execution engine");

            maybeEngine = std::move(*expectedEngine);
        }

        return *maybeEngine;
    }

    void Program::runJIT([[maybe_unused]] const std::optional<std::string> &objPath)
    {
        auto &engine = getOrCreateExecutionEngine();
        // FIXME: dtor of fn results in segfault
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
        auto start = std::chrono::high_resolution_clock::now();
        auto invocationResult = engine->invokePacked("main", params);
        auto stop = std::chrono::high_resolution_clock::now();
        float currentTime = float(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
        timer += currentTime;
        if (invocationResult)
        {
            throw JITInvocationError();
        }
    }

    void Program::convertToLLVM()
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
            /*optLevel=*/config.optimize ? 3 : 0, /*sizeLevel=*/0,
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

    void Program::lowerMLIR()
    {
        ::mlir::PassManager pm(&context);
        if (config.debug)
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

        // Partially lower voila to linalg
        optPM.addPass(::voila::mlir::createLowerToLinalgPass());
        // Partially lower voila to affine with a few cleanups
        optPM.addPass(::voila::mlir::createLowerToAffinePass());

        optPM.addPass(createCanonicalizerPass());
        optPM.addPass(createCSEPass());
        optPM.addPass(createConvertElementwiseToLinalgPass());
        optPM.addPass(createLinalgElementwiseOpFusionPass());
        optPM.addPass(createLinalgTilingPass());
        // bufferization passes
        // pm.addPass(createLinalgComprehensiveModuleBufferizePass());
        optPM.addPass(createSCFBufferizePass());
        optPM.addPass(createLinalgDetensorizePass());
        optPM.addPass(createLinalgBufferizePass());
        optPM.addPass(createStdBufferizePass());
        optPM.addPass(createTensorBufferizePass());
        pm.addPass(createTensorConstantBufferizePass());
        pm.addPass(createFuncBufferizePass());
        optPM.addPass(createCanonicalizerPass());
        optPM.addPass(createCSEPass());
        optPM.addPass(createBufferHoistingPass());
        pm.addPass(createNormalizeMemRefsPass());
        optPM.addPass(createPipelineDataTransferPass());
        auto state = pm.run(*mlirModule);
        spdlog::debug(MLIRModuleToString(mlirModule));

        if (failed(state))
        {
            throw MLIRLoweringError();
        }

        // FIXME: properly apply passes
        ::mlir::PassManager secondpm(&context);
        if (config.debug)
        {
            secondpm.enableStatistics();
            // secondpm.enableIRPrinting();
        }
        // Apply any generic pass manager command line options and run the pipeline.
        applyPassManagerCLOptions(secondpm);
        ::mlir::OpPassManager &secondOptPM = secondpm.nest<FuncOp>();

        secondOptPM.addPass(createConvertLinalgToParallelLoopsPass());
        secondOptPM.addPass(createConvertLinalgToLoopsPass());
        secondOptPM.addPass(createConvertLinalgToAffineLoopsPass());

        if (config.optimize)
        {
            secondOptPM.addPass(createPromoteBuffersToStackPass());
            secondOptPM.addPass(createSimplifyAffineStructuresPass());
            secondOptPM.addPass(createAffineLoopInvariantCodeMotionPass());
            secondOptPM.addPass(createAffineLoopNormalizePass());
            // secondOptPM.addPass(createLoopUnrollPass());
            secondOptPM.addPass(createAffineParallelizePass());
            secondOptPM.addPass(createAffineScalarReplacementPass());
            secondOptPM.addPass(createSuperVectorizePass(4));
            secondOptPM.addPass(createLowerAffinePass());
            secondOptPM.addPass(createParallelLoopSpecializationPass());
            secondOptPM.addPass(createParallelLoopFusionPass());
            secondOptPM.addPass(createParallelLoopTilingPass());
            secondOptPM.addPass(createLoopFusionPass());
            secondOptPM.addPass(createLoopCoalescingPass());
            secondOptPM.addPass(createLoopUnrollAndJamPass());
            secondpm.addPass(createAsyncParallelForPass());

            // secondpm.addPass(createBufferResultsToOutParamsPass());
            //  secondOptPM.addPass(createAffineDataCopyGenerationPass());
            //  secondOptPM.addPass(createLoopUnrollPass());
        }

        // secondOptPM.addPass(createCanonicalizerPass());
        // secondOptPM.addPass(createCSEPass());
        optPM.addPass(createFinalizingBufferizePass());
        secondpm.addPass(createNormalizeMemRefsPass());
        secondOptPM.addPass(createBufferDeallocationPass());

        secondpm.addPass(createStripDebugInfoPass());
        secondpm.addPass(createConvertVectorToLLVMPass());
        secondpm.addPass(createConvertAsyncToLLVMPass());
        secondOptPM.addPass(createLowerToCFGPass());
        secondpm.addPass(::voila::mlir::createLowerToLLVMPass());
        secondpm.addPass(createMemRefToLLVMPass());
        secondOptPM.addPass(createCanonicalizerPass());
        secondOptPM.addPass(createCSEPass());

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
        if (config.debug)
        {
            // context.disableMultithreading(); //FIXME: with threading disabled, the program segfaults
        }
        try
        {
            mlirModule = ::voila::MLIRGenerator::mlirGen(context, *this);
        }
        catch (MLIRModuleVerificationError &err)
        {
            spdlog::error(MLIRModuleToString(mlirModule));
            throw err;
        }
        if (!mlirModule)
        {
            spdlog::error(MLIRModuleToString(mlirModule));
            throw ::voila::MLIRGenerationError();
        }
        spdlog::debug(MLIRModuleToString(mlirModule));
        return mlirModule;
    }

    Program &Program::operator<<(Parameter param)
    {
        const auto &main = functions.at("main");
        Expression arg;
        try
        {
            arg = main->args.at(nparam++);
        }
        catch (std::out_of_range &ex)
        {
            throw ArgsOutOfRangeError();
        }
        assert(arg.is_variable());
        inferer.insertNewType(*arg.as_variable(), param.type, Arity(param.size));

        assert(param.type != DataType::NUMERIC && param.type != DataType::VOID && param.type != DataType::UNKNOWN);

        if (param.size == 0) // scalar type
        {
            params.push_back(param.data);
        }
        else
        {
            // store strided 1D memref as unpacked struct defined in
            // llvm/mlir/include/mlir/ExecutionEngine/CRunnerUtils.h
            // use malloc to have matching alloc-dealloc while being able to use free on void *
            void **ptr = static_cast<void **>(std::malloc(sizeof(void *)));
            *ptr = param.data;
            params.push_back(ptr);
            toDealloc.push_back(params.back());
            params.push_back(ptr); // base ptr
            auto *tmp = static_cast<int64_t *>(std::malloc(sizeof(int64_t)));
            *tmp = 0;
            params.push_back(tmp); // offset
            toDealloc.push_back(params.back());
            tmp = static_cast<int64_t *>(std::malloc(sizeof(int64_t)));
            *tmp = 0;
            params.push_back(tmp); // sizes
            toDealloc.push_back(params.back());
            tmp = static_cast<int64_t *>(std::malloc(sizeof(int64_t)));
            *tmp = 1;
            params.push_back(tmp); // strides
            toDealloc.push_back(params.back());
        }

        return *this;
    }

    // either return void, scalars or pointer to strided memref types as unique_ptr
    std::vector<Program::result_t> Program::operator()()
    {
        if (!maybeEngine)
        {
            if (config.plotAST)
            {
                to_dot(config.ASTOutFile);
            }

            // generate mlir
            spdlog::debug("Start type inference");
            inferTypes();
            spdlog::debug("Finished type inference");

            spdlog::debug("Start mlir generation");
            // generate mlir
            generateMLIR();
            spdlog::debug("Finished mlir generation");
            if (config.printMLIR)
            {
                printMLIR(config.MLIROutFile);
            }
            spdlog::debug("Start mlir lowering");
            // lower mlir
            lowerMLIR();
            spdlog::debug("Finished mlir lowering");
            if (config.printLoweredMLIR)
            {
                printMLIR(config.MLIRLoweredOutFile);
            }
            spdlog::debug("Start mlir to llvm conversion");
            // lower to llvm
            convertToLLVM();
            if (config.printLLVM)
            {
                printLLVM(config.LLVMOutFile);
            }
            spdlog::debug("Finished mlir to llvm conversion");
        }

        // run jit
        const auto &main = functions.at("main");
        std::vector<result_t> res;
        if (main->result.has_value())
        {
            auto &type = inferer.get_type(main->result.value());
            // test scalar

            SmallVector<::llvm::Type *> resTypes;

            auto types = type.getTypes();
            auto arities = type.getArities();
            for (size_t i = 0; i < types.size(); ++i)
            {
                switch (types[i])
                {
                    case DataType::BOOL: // FIXME: I1 type is not directly mappable to C++ types
                        throw NotImplementedException();
                    case DataType::NUMERIC:
                        throw std::logic_error("Abstract type");
                    case DataType::INT32:
                        if (!arities[i].is_undef() && arities[i].get_size() == 0)
                        {
                            resTypes.push_back(::llvm::Type::getInt32Ty(llvmContext));
                        }
                        else
                        {
                            resTypes.push_back(llvm::StructType::create(
                                llvmContext,
                                {llvm::Type::getInt32PtrTy(llvmContext), llvm::Type::getInt32PtrTy(llvmContext),
                                 ::llvm::Type::getInt64Ty(llvmContext), ::llvm::Type::getInt64Ty(llvmContext),
                                 ::llvm::Type::getInt64Ty(llvmContext)}));
                        }
                        break;
                    case DataType::INT64:
                        if (!arities[i].is_undef() && arities[i].get_size() == 0)
                        {
                            resTypes.push_back(::llvm::Type::getInt64Ty(llvmContext));
                        }
                        else
                        {
                            resTypes.push_back(llvm::StructType::create(
                                llvmContext,
                                {llvm::Type::getInt64PtrTy(llvmContext), llvm::Type::getInt64PtrTy(llvmContext),
                                 ::llvm::Type::getInt64Ty(llvmContext), ::llvm::Type::getInt64Ty(llvmContext),
                                 ::llvm::Type::getInt64Ty(llvmContext)}));
                        }
                        break;
                    case DataType::DBL:
                        if (!arities[i].is_undef() && arities[i].get_size() == 0)
                        {
                            resTypes.push_back(::llvm::Type::getDoubleTy(llvmContext));
                        }
                        else
                        {
                            resTypes.push_back(llvm::StructType::create(
                                llvmContext,
                                {llvm::Type::getDoublePtrTy(llvmContext), llvm::Type::getDoublePtrTy(llvmContext),
                                 ::llvm::Type::getInt64Ty(llvmContext), ::llvm::Type::getInt64Ty(llvmContext),
                                 ::llvm::Type::getInt64Ty(llvmContext)}));
                        }
                        break;
                    case DataType::STRING:
                        throw NotImplementedException();
                    case DataType::VOID:
                        break;
                    case DataType::UNKNOWN:
                        throw std::logic_error("");
                }
            }

            ::llvm::DataLayout layout(llvmModule.get());
            auto resStruct = llvm::StructType::create(llvmContext, resTypes);
            auto resLayout = layout.getStructLayout(resStruct);
            params.push_back(std::malloc(resLayout->getSizeInBytes()));
            spdlog::debug("Running JIT Program");
            runJIT();
            spdlog::debug("Finished Running JIT Program");
            std::shared_ptr<void> basePtr(
                params.pop_back_val(),
                ProgramResultDeleter());

            for (size_t i = 0; i < types.size(); ++i)
            {
                switch (types[i])
                {
                    case DataType::INT32:
                        if (!arities[i].is_undef() && arities[i].get_size() == 0)
                        {
                            auto *extractedRes = reinterpret_cast<uint32_t *>(
                                std::reinterpret_pointer_cast<char>(basePtr).get() + resLayout->getElementOffset(i));
                            res.emplace_back(*extractedRes);
                        }
                        else
                        {
                            res.emplace_back(strided_memref_ptr<uint32_t, 1>(
                                basePtr, reinterpret_cast<StridedMemRefType<uint32_t, 1> *>(
                                             std::reinterpret_pointer_cast<char>(basePtr).get() +
                                             resLayout->getElementOffset(i))));
                            std::get_deleter<ProgramResultDeleter>(basePtr)->toDealloc.push_back(
                                std::get<strided_memref_ptr<uint32_t, 1>>(res.back())->basePtr);
                        }
                        break;
                    case DataType::INT64:
                        if (!arities[i].is_undef() && arities[i].get_size() == 0)
                        {
                            auto *extractedRes = reinterpret_cast<uint64_t *>(
                                std::reinterpret_pointer_cast<char>(basePtr).get() + resLayout->getElementOffset(i));
                            res.emplace_back(*extractedRes);
                        }
                        else
                        {
                            res.emplace_back(strided_memref_ptr<uint64_t, 1>(
                                basePtr, reinterpret_cast<StridedMemRefType<uint64_t, 1> *>(
                                             std::reinterpret_pointer_cast<char>(basePtr).get() +
                                             resLayout->getElementOffset(i))));
                            std::get_deleter<ProgramResultDeleter>(basePtr)->toDealloc.push_back(
                                std::get<strided_memref_ptr<uint64_t, 1>>(res.back())->basePtr);
                            /*                            std::get_deleter<ProgramResultDeleter
                               *>(basePtr)->toDealloc.push_back( std::get<strided_memref_ptr<uint64_t,
                               1>>(res.back())->basePtr);*/
                        }
                        break;
                    case DataType::DBL:
                        if (!arities[i].is_undef() && arities[i].get_size() == 0)
                        {
                            auto *extractedRes = reinterpret_cast<double *>(
                                std::reinterpret_pointer_cast<char>(basePtr).get() + resLayout->getElementOffset(i));
                            res.emplace_back(*extractedRes);
                        }
                        else
                        {
                            res.emplace_back(strided_memref_ptr<double, 1>(
                                basePtr, reinterpret_cast<StridedMemRefType<double, 1> *>(
                                             std::reinterpret_pointer_cast<char>(basePtr).get() +
                                             resLayout->getElementOffset(i))));
                            std::get_deleter<ProgramResultDeleter>(basePtr)->toDealloc.push_back(
                                std::get<strided_memref_ptr<double, 1>>(res.back())->basePtr);
                        }
                        break;
                    default:
                        throw std::logic_error("");
                }
            }
        }
        params.clear();
        for (auto elem : toDealloc)
        {
            std::free(elem);
        }
        toDealloc.clear();
        return res;
    }

    void Program::add_func(ast::Fun *f)
    {
        functions.emplace(f->name, f);
        f->variables = std::move(func_vars);
        func_vars.clear();
    }

    void Program::inferTypes()
    {
        TypeInferencePass typeInference(inferer);
        typeInference.inferTypes(*this);
    }

    Program::Program(std::string_view source_path, Config config) :
        func_vars(),
        context(),
        llvmContext(),
        mlirModule(),
        llvmModule(),
        maybeEngine(std::nullopt),
        functions(),
        config{std::move(config)},
        inferer()
    {
        std::ifstream fst(source_path.data(), std::ios::in);

        if (fst.is_open())
        {
            lexer = new Lexer(fst);        // read file, decode UTF-8/16/32 format
            lexer->filename = source_path; // the filename to display with error locations

            ::voila::parser::Parser parser(*lexer, *this);
            if (parser() != 0)
                throw ::voila::ParsingError();
        }
        else
        {
            spdlog::error("failed to open {}", source_path);
        }
    }

    Program::Program(Config config) :
        func_vars(),
        context(),
        llvmContext(),
        mlirModule(),
        llvmModule(),
        maybeEngine(std::nullopt),
        functions(),
        config{std::move(config)},
        lexer{new Lexer()},
        inferer()
    {
    }

    /**
     * @param data pointer to data
     * @param size  number of elements in @link{data} pointer, 0 if pointer to scalar
     * @param type data type compatible to values in @link{data}
     * @return
     */
    Parameter make_param(void *data, size_t size, DataType type)
    {
        return Parameter(data, size, type);
    }
} // namespace voila
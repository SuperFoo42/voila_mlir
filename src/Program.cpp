#include "Program.hpp"
#include "ArgsOutOfRangeError.hpp"
#include "Config.hpp"
#include "DotVisualizer.hpp"
#include "JITInvocationError.hpp"
#include "LLVMGenerationError.hpp"
#include "MLIRGenerationError.hpp"
#include "MLIRGenerator.hpp"
#include "MLIRLoweringError.hpp"
#include "NotImplementedException.hpp"
#include "ParsingError.hpp"
#include "TypeInferencePass.hpp"
#include "TypeInferer.hpp"
#include "ast/ASTNode.hpp"
#include "ast/ASTVisitor.hpp"
#include "ast/Fun.hpp"
#include "ast/Variable.hpp"
#include "defs.hpp"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialects/Voila/IR/VoilaOpsDialect.h.inc"
#include "mlir/Dialects/Voila/Passes/MemOpsToAffineMemOpsConversionPass.hpp"
#include "mlir/Dialects/Voila/Passes/VoilaBufferizePass.hpp"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "range/v3/iterator/basic_iterator.hpp"
#include "range/v3/range_fwd.hpp"
#include "range/v3/view/adaptor.hpp"
#include "range/v3/view/facade.hpp"
#include "voila_lexer.hpp"
#include "voila_parser.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_os_ostream.h"
#include <MlirModuleVerificationError.hpp>
#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <llvm/IR/AssemblyAnnotationWriter.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialects/Voila/Passes/ParallelLoopToGpuMappingPass.hpp>
#include <mlir/Dialects/Voila/Passes/PredicationForwardingPass.hpp>
#include <mlir/Dialects/Voila/Passes/VoilaToAffineLoweringPass.hpp>
#include <mlir/Dialects/Voila/Passes/VoilaToLLVMLoweringPass.hpp>
#include <mlir/Dialects/Voila/Passes/VoilaToLinalgLoweringPass.hpp>
#include <mlir/ExecutionEngine/CRunnerUtils.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>
#include <mlir/Transforms/Passes.h>
#include <stdexcept>
#include <system_error>
#include <type_traits>
#include <utility>

namespace voila
{
    using namespace ast;
    using namespace ::mlir;
    using namespace func;
    using namespace ::voila::lexer;
    using namespace ::voila::mlir;
    using namespace ::mlir::arith;

    template <typename T>
    static auto &resolveAndFetchResult(Arity arity,
                                       std::shared_ptr<void> &basePtr,
                                       void *elemPtr,
                                       std::vector<Program::result_t> &res)
    {
        if (!arity.undef() && arity.get_size() <= 1)
        {
            auto *extractedRes = reinterpret_cast<T *>(elemPtr);
            res.emplace_back(*extractedRes);
        }
        else
        {
            res.emplace_back(strided_memref_ptr<T, 1>(basePtr, reinterpret_cast<StridedMemRefType<T, 1> *>(elemPtr)));
            std::get_deleter<ProgramResultDeleter>(basePtr)->toDealloc.push_back(
                std::get<strided_memref_ptr<T, 1>>(res.back())->basePtr);
        }

        return res;
    }

    static std::string MLIRModuleToString(OwningOpRef<ModuleOp> &module)
    {
        std::string res;
        llvm::raw_string_ostream os(res);
        module->print(os);
        os.flush();
        return res;
    }

    void Program::add_func(ast::ASTNodeVariant f)
    {
        std::visit(ast::overloaded{[&](std::shared_ptr<ast::Fun> &fun)
                                   {
                                       functions.emplace(fun->name(), fun);
                                       fun->variables() = std::move(func_vars);
                                       func_vars.clear();
                                   },
                                   [&](std::shared_ptr<ast::Main> &m)
                                   {
                                       functions.emplace(m->name(), m);
                                       m->variables() = std::move(func_vars);
                                       func_vars.clear();
                                   },
                                   [](auto ) { throw std::logic_error("Invalid type"); }},
                   f);
    }


    void Program::add_cloned_func(ast::ASTNodeVariant f)
    {
        std::visit(ast::overloaded{[&](std::shared_ptr<ast::Fun> &fun)
                                   {
                                       fun->variables() = std::move(func_vars);
                                       functions.emplace(fun->name() + "_" + to_string(*inferer.get_type(fun)), fun);
                                       func_vars.clear();
                                   },
                                   [&](std::shared_ptr<ast::Main> &m)
                                   {
                                       auto fun = std::dynamic_pointer_cast<ast::Fun>(m);
                                       fun->variables() = std::move(func_vars);
                                       functions.emplace(fun->name() + "_" + to_string(*inferer.get_type(fun)), fun);
                                       func_vars.clear();
                                   },
                                   [](auto ) { throw std::logic_error("Invalid type"); }},
                   f);
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

    /*    void Program::infer_type(Expression &node) {
            node.visit<decltype(inferer), decltype(inferer)::result_type>(inferer);
        }

        void Program::infer_type(AbstractASTNode &node) {
            inferer(&node);
        }

        void Program::infer_type(Statement &node) {
            node.visit<decltype(inferer), decltype(inferer)::result_type>(inferer);
        }*/

    void Program::to_dot(const std::string &fname = "-")
    {
        for (auto &func : life_funcs())
        {
            DotVisualizer vis(func, std::optional<std::reference_wrapper<TypeInferer>>(inferer));

            auto out = std::unique_ptr<std::ostream>(
                fname.empty() || fname == "-" ? &std::cout : new std::ofstream(func->name() + fname, std::ios::out));

            *out << vis;
        }
    }

    bool Program::has_var(const std::string &var_name) { return func_vars.contains(var_name); }

    ASTNodeVariant Program::get_var(const std::string &var_name) { return func_vars.at(var_name); }

    void Program::add_var(const ASTNodeVariant &expr)
    {
        auto var = std::get<std::shared_ptr<Variable>>(expr);

        func_vars.emplace(var->var, expr);
    }

    const ::mlir::OwningOpRef<ModuleOp> &Program::getMLIRModule() const { return mlirModule; }

    [[maybe_unused]] const ::mlir::MLIRContext &Program::getMLIRContext() const { return context; }

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
            auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
            if (!tmBuilderOrError)
                throw std::runtime_error("Failed to create a JITTargetMachineBuilder for the host");

            auto tmOrError = tmBuilderOrError->createTargetMachine();
            if (!tmOrError)
                throw std::runtime_error("Failed to create a TargetMachine for the host");

            auto optPipeline = makeOptimizingTransformer(
                /*optLevel=*/config._optimize ? llvm::CodeGenOpt::Aggressive : llvm::CodeGenOpt::None,
                /*sizeLevel=*/llvm::CodeModel::Tiny,
                /*targetMachine=*/tmOrError->get());

            SmallVector<StringRef, 4> executionEngineLibs;
            SmallVector<SmallString<255>, 4> libPaths(
                {SmallString<255>(MLIR_UTILS_LIB), SmallString<255>(MLIR_C_UTILS_LIB)});
            if (config._async_parallel)
                libPaths.emplace_back(MLIR_ASYNC_LIB);
            if (config._openmp_parallel)
                libPaths.emplace_back(MLIR_OPENMP_LIB);

            using MlirRunnerInitFn = void (*)(llvm::StringMap<void *> &);
            using MlirRunnerDestroyFn = void (*)();

            llvm::StringMap<void *> exportSymbols;
            SmallVector<MlirRunnerDestroyFn> destroyFns;

            // Handle libraries that do support mlir-runner init/destroy callbacks.
            // TODO: only search libs required by config for faster compile-time
            for (auto &libPath : libPaths)
            {
                auto lib = llvm::sys::DynamicLibrary::getPermanentLibrary(libPath.c_str());
                void *initSym = lib.getAddressOfSymbol("__mlir_runner_init");
                void *destroySim = lib.getAddressOfSymbol("__mlir_runner_destroy");

                // Library does not support mlir runner, load it with ExecutionEngine.
                if (!initSym || !destroySim)
                {
                    executionEngineLibs.push_back(libPath);
                    continue;
                }

                auto initFn = reinterpret_cast<MlirRunnerInitFn>(initSym);
                initFn(exportSymbols);

                auto destroyFn = reinterpret_cast<MlirRunnerDestroyFn>(destroySim);
                destroyFns.push_back(destroyFn);
            }

            // Build a runtime symbol map from the config and exported symbols.
            /*            auto runtimeSymbolMap = [&](llvm::orc::MangleAndInterner interner) {
                            auto symbolMap = llvm::orc::SymbolMap();
                            for (auto &exportSymbol: exportSymbols)
                                symbolMap[interner(exportSymbol.getKey())] =
                                        llvm::JITEvaluatedSymbol::fromPointer(exportSymbol.getValue());
                            return symbolMap;
                        };*/

            // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
            // the module.
            ExecutionEngineOptions opts;
            opts.transformer = optPipeline;
            // opts.enableObjectCache = true;
            opts.jitCodeGenOptLevel = llvm::CodeGenOpt::Level::Aggressive;
            opts.enableObjectDump = true;
            SmallVector<StringRef> pathRefs;
            for (auto &path : libPaths)
                pathRefs.push_back(path);
            opts.sharedLibPaths = pathRefs;
            auto expectedEngine = ExecutionEngine::create(*mlirModule, opts);
            assert(expectedEngine && "failed to construct an execution engine");

            maybeEngine = std::move(*expectedEngine);
            //(*maybeEngine)->registerSymbols(runtimeSymbolMap);
        }

        return *maybeEngine;
    }

    void Program::runJIT([[maybe_unused]] const std::optional<std::string> &objPath)
    {
        auto &engine = getOrCreateExecutionEngine();

        // Invoke the JIT-compiled function.
        /*        Profiler<*//*Events::L3_CACHE_MISSES, Events::L2_CACHE_MISSES, Events::BRANCH_MISSES, Events::TLB_MISSES,
                Events::NO_INST_COMPLETE, Events::CY_STALLED, Events::REF_CYCLES, *//*Events::TOT_CYCLES,
                Events::INS_ISSUED, Events::PREFETCH_MISS>
                prof;
        prof.start();*/
        auto invocationResult = engine->invokePacked("main", params);
        /*        prof.stop();
                //
                if (config._debug) {
                    std::cout << prof << std::endl;
                }
                if (config._profile) {
                    timer = prof.getTime();
                    // TODO: store profiling results
                }*/

        if (invocationResult)
        {
            throw JITInvocationError();
        }
        if (objPath)
            engine->dumpToObjectFile(*objPath);
    }

    void Program::convertToLLVM()
    {
        // lower to llvm
        if (config._openmp_parallel)
        {
            registerOpenMPDialectTranslation(*mlirModule->getContext());
        }
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
        //TODO: options
        //ExecutionEngine::setupTargetTriple(llvmModule.get());
        // An optimization pipeline to use within the execution engine.
        auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
        if (!tmBuilderOrError)
            throw std::runtime_error("Failed to create a JITTargetMachineBuilder for the host");

        auto tmOrError = tmBuilderOrError->createTargetMachine();
        if (!tmOrError)
            throw std::runtime_error("Failed to create a TargetMachine for the host");

        auto optPipeline = makeOptimizingTransformer(
            /*optLevel=*/config._optimize ? llvm::CodeGenOpt::Aggressive : llvm::CodeGenOpt::None,
            /*sizeLevel=*/llvm::CodeModel::Tiny,
            /*targetMachine=*/tmOrError->get());
        if (auto err = optPipeline(llvmModule.get()))
        {
            throw err; // TODO: rethrow with other exception
        }
        // TODO:

        llvm::dbgs() << LLVMModuleToString(*llvmModule);
    }

    void Program::printLLVM(const std::string &filename)
    {
        std::error_code ec;
        llvm::raw_fd_ostream os(filename + ".llvm", ec, llvm::sys::fs::OF_None);
        llvm::AssemblyAnnotationWriter aw;
        llvmModule->print(os, &aw);
        os.flush();
    }

    void Program::printMLIR(const std::string &filename = "-")
    {
        std::error_code ec;
        auto os = std::unique_ptr<llvm::raw_ostream>(
            (filename.empty() || filename == "-")
                ? static_cast<llvm::raw_ostream *>(new llvm::raw_os_ostream(std::cout))
                : static_cast<llvm::raw_ostream *>(
                      new llvm::raw_fd_ostream(filename + ".mlir", ec, llvm::sys::fs::OF_None)));

        mlirModule->print(*os);
        os->flush();
    }

    void Program::lowerMLIR()
    {
        ::mlir::PassManager pm(&context);
        if (config._debug)
        {
            pm.enableStatistics();
        }
        // Apply any generic pass manager command line options and run the pipeline.
        applyPassManagerCLOptions(pm);

        pm.addPass(createInlinerPass());
        // Now that there is only one function, we can infer the shapes of each of
        // the operations.
        pm.addNestedPass<FuncOp>(createCanonicalizerPass());
        pm.addNestedPass<FuncOp>(createCSEPass());
        if (config._optimize && config._optimize_selections)
            pm.addNestedPass<FuncOp>(createPredicationForwardingPass());

        // Partially lower voila to linalg
        pm.addNestedPass<FuncOp>(createLowerToLinalgPass());
        // Partially lower voila to affine with a few cleanups
        pm.addNestedPass<FuncOp>(createLowerToAffinePass());

        pm.addNestedPass<FuncOp>(createCanonicalizerPass());
        pm.addNestedPass<FuncOp>(createCSEPass());

        pm.addNestedPass<FuncOp>(createConvertElementwiseToLinalgPass());
        // pm.addNestedPass<FuncOp>(createLinalgStrategyEnablePass());
        //  optPM.addPass(createLinalgGeneralizationPass());
        if (config._optimize && config._fuse)
            pm.addNestedPass<FuncOp>(createLinalgElementwiseOpFusionPass());
        // pm.addNestedPass<FuncOp>(createLinalgStrategyPromotePass());

        // pm.addNestedPass<FuncOp>(mlir::createLinalgTiledPeelingPass());

        /*if (config._optimize && config._tile)
        {
            auto tile_size = {config._tile_size == -1 ? max_in_table_size / config._parallel_threads :
                                                        config._tile_size};
            //            if (config._peel)
            //            {
*//*            pm.addNestedPass<FuncOp>(
                createLinalgTilingPass(tile_size, ::mlir::linalg::LinalgTilingLoopType::TiledLoops, {}, {0}));*//*
            //            }
            //            else
            //            {
            //                pm.addNestedPass<FuncOp>(createLinalgTilingPass(
            //                    tile_size, ::mlir::linalg::LinalgTilingLoopType::TiledLoops,
            //                    {"CyclicNumProcsEqNumIters"}));
            //            }

*//*            pm.addNestedPass<FuncOp>(
                            createLinalgStrategyTileAndFusePass(""
                                                                ,linalg::LinalgTilingAndFusionOptions{tile_size,
               {}}));*//*

        }*/
        // pm.addNestedPass<FuncOp>(createLinalgTilingPass(10, linalg::LinalgTilingLoopType::AffineLoops));
        auto o = linalg::LinalgTilingAndFusionOptions();
        o.setTileSizes({10});
        // pm.addNestedPass<FuncOp>(createLinalgStrategyTileAndFusePass(llvm::StringRef(""),o));
        // pm.addNestedPass<FuncOp>(createLinalgStrategyVectorizePass());

        //  bufferization passes
        pm.addPass(createVoilaBufferizePass());
        pm.addNestedPass<FuncOp>(tensor::createTensorBufferizePass());
        pm.addNestedPass<FuncOp>(createLinalgBufferizePass());
        // FIXME: pm.addPass(createFuncBufferizePass());

        pm.addNestedPass<FuncOp>(bufferization::createBufferHoistingPass());
        pm.addNestedPass<FuncOp>(bufferization::createBufferLoopHoistingPass());
        // pm.addNestedPass<FuncOp>(bufferization::createPromoteBuffersToStackPass()); //potential double free if result
        // res small enough to fit into stack
        pm.addNestedPass<FuncOp>(affine::createSimplifyAffineStructuresPass());
        pm.addNestedPass<FuncOp>(affine::createAffineScalarReplacementPass());
        pm.addNestedPass<FuncOp>(affine::createAffineLoopInvariantCodeMotionPass());
        pm.addNestedPass<FuncOp>(affine::createAffineLoopNormalizePass());
        pm.addNestedPass<FuncOp>(createCanonicalizerPass());
        pm.addNestedPass<FuncOp>(createCSEPass());

        pm.addNestedPass<FuncOp>(createConvertLinalgToAffineLoopsPass());
        if (config._optimize && config._vectorize)
        {
            std::unique_ptr<Pass> vectorizationPass = affine::createAffineVectorize();
            if (config._vectorize_reductions)
            {
                (void)vectorizationPass->initializeOptions("vectorize-reductions=true");
            }
            (void)vectorizationPass->initializeOptions("test-fastest-varying=0");
            pm.addNestedPass<FuncOp>(std::move(vectorizationPass));
        }
        // pm.addNestedPass<FuncOp>(::voila::mlir::createConvertLinalgTiledLoopsToAffineForPass());

        pm.addNestedPass<FuncOp>(affine::createSimplifyAffineStructuresPass());
        pm.addNestedPass<FuncOp>(affine::createAffineScalarReplacementPass());
        pm.addNestedPass<FuncOp>(affine::createAffineLoopInvariantCodeMotionPass());
        pm.addNestedPass<FuncOp>(affine::createAffineLoopNormalizePass());
        pm.addNestedPass<FuncOp>(createMemOpsToAffineMemOpsConversionPass());
        pm.addNestedPass<FuncOp>(createCanonicalizerPass());
        pm.addNestedPass<FuncOp>(createCSEPass());
        //TODO
        //pm.addNestedPass<FuncOp>(affine::createLoopFusionPass(0, 0, true));

        // pm.addNestedPass<FuncOp>(createConvertLinalgTiledLoopsToSCFPass());
        //
        // pm.addNestedPass<FuncOp>(createLinalgStrategyVectorizePass());
        //         pm.addNestedPass<FuncOp>(createConvertLinalgToAffineLoopsPass());
        //  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
        //  pm.addNestedPass<FuncOp>(createConvertLinalgToAffineLoopsPass());

        // TODO: pm.addNestedPass<FuncOp>(createConvertLinalgTiledLoopsToAffineForPass());
        if (config._optimize && config._parallelize)
        {
            pm.addNestedPass<FuncOp>(createConvertLinalgToParallelLoopsPass());
        }

        pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());

        pm.addNestedPass<FuncOp>(affine::createSimplifyAffineStructuresPass());
        pm.addNestedPass<FuncOp>(affine::createAffineScalarReplacementPass());
        pm.addNestedPass<FuncOp>(affine::createAffineLoopInvariantCodeMotionPass());
        pm.addNestedPass<FuncOp>(affine::createAffineLoopNormalizePass());
        pm.addNestedPass<FuncOp>(createCanonicalizerPass());
        pm.addNestedPass<FuncOp>(createCSEPass());

        if (config._optimize && config._parallelize)
        {
            std::unique_ptr<Pass> parallelizationPass = affine::createAffineParallelizePass();
            if (config._parallelize_reductions)
            {
                (void)parallelizationPass->initializeOptions("parallel-reductions=true");
            }
            pm.addNestedPass<FuncOp>(std::move(parallelizationPass));
        }
        pm.addNestedPass<FuncOp>(createLoopInvariantCodeMotionPass());
        // pm.addNestedPass<FuncOp>(createAffineForToGPUPass());
        // pm.addNestedPass<FuncOp>(createLoopTilingPass(65000));
        // pm.addNestedPass<FuncOp>(createLoopCoalescingPass());
        // pm.addNestedPass<FuncOp>(createForLoopPeelingPass());
        // pm.addNestedPass<FuncOp>(createLoopUnrollPass(8));

        if (config._optimize && config._unroll_factor > 1)
            pm.addNestedPass<FuncOp>(affine::createLoopUnrollAndJamPass(config._unroll_factor));
        pm.addNestedPass<FuncOp>(createForLoopSpecializationPass());
        pm.addNestedPass<FuncOp>(createParallelLoopFusionPass());
        //TODO
        //pm.addNestedPass<FuncOp>(createParallelLoopCollapsingPass());
        pm.addNestedPass<FuncOp>(createParallelLoopTilingPass());
        pm.addNestedPass<FuncOp>(createParallelLoopSpecializationPass());
        pm.addNestedPass<FuncOp>(createLowerAffinePass());
        if (config._optimize && config._async_parallel)
        {
            pm.addPass(createAsyncParallelForPass(true, config._parallel_threads, 1));
            pm.addPass(createAsyncToAsyncRuntimePass());
            pm.addPass(createAsyncRuntimeRefCountingPass());
            pm.addPass(createAsyncRuntimeRefCountingOptPass());
        }
        if (config._optimize && config._openmp_parallel)
        {
            pm.addPass(createConvertSCFToOpenMPPass());
        }

        if (config._optimize && config._gpu_parallel)
        {
            pm.addNestedPass<FuncOp>(createConvertVectorToGPUPass());
            pm.addNestedPass<FuncOp>(createParallelLoopToGPUMappingPass());
            pm.addNestedPass<FuncOp>(createParallelLoopToGpuPass());
        }

        //TODO
//        if (config._optimize && config._fuse)
//            pm.addNestedPass<FuncOp>(createLoopFusionPass());

        pm.addNestedPass<FuncOp>(createCanonicalizerPass());
        pm.addNestedPass<FuncOp>(createCSEPass());
        pm.addPass(createConstantBufferizePass());
        pm.addNestedPass<FuncOp>(bufferization::createBufferDeallocationPass());
        pm.addNestedPass<FuncOp>(::mlir::bufferization::createFinalizingBufferizePass());

        if (config._optimize && config._vectorize)
        {
            // TODO
            /*            pm.addNestedPass<FuncOp>(createLinalgStrategyLowerVectorsPass(
                                linalg::LinalgVectorizationOptions()
                                        .enableContractionLowering()
                                        .enableMultiReductionLowering()
                                        .enableTransferLowering()
                                        .enableTransferToSCFConversion()
                                        .enableTransferPartialRewrite()
                                        .setVectorTransferToSCFOptions(VectorTransferToSCFOptions().enableFullUnroll())
                                        .setVectorTransformsOptions(
                                                vector::VectorTransformsOptions().setVectorMultiReductionLowering(
                                                        vector::VectorMultiReductionLowering::InnerReduction))));
            pm.addPass(createConvertVectorToLLVMPass(LowerVectorToLLVMOptions()
                                                         .enableIndexOptimizations(true)
                                                         .enableReassociateFPReductions(true)
                                                         .enableX86Vector(true)));
                                                         */
        }

        pm.addNestedPass<FuncOp>(createArithExpandOpsPass());
        pm.addNestedPass<FuncOp>(memref::createExpandOpsPass());
        // pm.addPass(createLowerHostCodeToLLVMPass());
        pm.addNestedPass<FuncOp>(createLowerAffinePass());

        if (config._optimize && config._async_parallel)
        {
            pm.addPass(createConvertAsyncToLLVMPass());
        }
        if (config._optimize && config._openmp_parallel)
        {
            pm.addPass(createConvertOpenMPToLLVMPass());
        }

        pm.addPass(createLowerVoilaToLLVMPass());
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
        pm.addPass(LLVM::createLegalizeForExportPass());

        auto state = pm.run(*mlirModule);
        llvm::dbgs() << MLIRModuleToString(mlirModule);
        if (failed(state))
        {
            throw MLIRLoweringError();
        }
    }

    OwningOpRef<ModuleOp> &Program::generateMLIR()
    {
        // Load our Dialect in this MLIR Context.
        context.getOrLoadDialect<::mlir::voila::VoilaDialect>();
        context.getOrLoadDialect<::mlir::func::FuncDialect>();
        if (config._debug)
        {
            // context.disableMultithreading(); //FIXME: with threading disabled, the program segfaults
        }
        try
        {
            mlirModule = ::voila::MLIRGenerator::mlirGen(context, *this);
        }
        catch (MLIRModuleVerificationError &err)
        {
            llvm::errs() << MLIRModuleToString(mlirModule);
            throw err;
        }
        if (!mlirModule)
        {
            llvm::errs() << MLIRModuleToString(mlirModule);
            throw ::voila::MLIRGenerationError();
        }
        llvm::dbgs() << MLIRModuleToString(mlirModule);
        return mlirModule;
    }

    Program &Program::operator<<(Parameter param)
    {
        const auto &main = functions.at("main");
        ASTNodeVariant arg;
        try
        {
            arg = main->args().at(nparam++);
        }
        catch (std::out_of_range &ex)
        {
            throw ArgsOutOfRangeError();
        }
        assert(std::holds_alternative<std::shared_ptr<Variable>>(arg));
        inferer.insertNewType(arg, param.type, Arity(param.size));

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
            *tmp = param.size;
            max_in_table_size = std::max(max_in_table_size, param.size);
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
        const auto &main = functions.at("main");
        if (nparam < main->args().size())
        {
            throw std::runtime_error("Not enough parameters supplied");
        }

        if (!maybeEngine)
        {
            if (config._plotAST)
            {
                to_dot(config.ASTOutFile);
            }

            // generate mlir
            llvm::dbgs() << "Start type inference";
            inferTypes();
            llvm::dbgs() << "Finished type inference";

            llvm::dbgs() << "Start mlir generation";
            // generate mlir
            generateMLIR();
            llvm::dbgs() << "Finished mlir generation";
            if (config._printMLIR)
            {
                printMLIR(config.MLIROutFile);
            }
            llvm::dbgs() << "Start mlir lowering";
            // lower mlir
            lowerMLIR();
            llvm::dbgs() << "Finished mlir lowering";
            if (config._printLoweredMLIR)
            {
                printMLIR(config.MLIRLoweredOutFile);
            }
            llvm::dbgs() << "Start mlir to llvm conversion";
            // lower to llvm
            convertToLLVM();
            if (config._printLLVM)
            {
                printLLVM(config.LLVMOutFile);
            }
            llvm::dbgs() << "Finished mlir to llvm conversion";
        }

        // run jit
        std::vector<result_t> res;
        if (std::holds_alternative<std::monostate>(main->result()))
        {
            auto type = inferer.get_type(main->result());
            // test scalar

            SmallVector<::llvm::Type *> resTypes;

            const auto types = type->getTypes();
            const auto arities = type->getArities();
            for (size_t i = 0; i < types.size(); ++i)
            {
                if (!arities[i].get().undef() && arities[i].get().get_size() <= 1)
                {
                    switch (types[i])
                    {
                    case DataType::INT32:
                        resTypes.push_back(::llvm::Type::getInt32Ty(llvmContext));
                        break;
                    case DataType::INT64:
                        resTypes.push_back(::llvm::Type::getInt64Ty(llvmContext));
                        break;
                    case DataType::DBL:
                        resTypes.push_back(::llvm::Type::getDoubleTy(llvmContext));
                        break;
                    case DataType::BOOL:
                    case DataType::STRING:
                        throw NotImplementedException();
                    case DataType::NUMERIC:
                    case DataType::VOID:
                        break;
                    case DataType::UNKNOWN:
                        throw std::logic_error("");
                    }
                }
                else
                {
                    switch (types[i])
                    {
                    case DataType::INT32:
                        resTypes.push_back(llvm::StructType::create(
                            llvmContext,
                            {llvm::Type::getInt8PtrTy(llvmContext), llvm::Type::getInt8PtrTy(llvmContext),
                             ::llvm::Type::getInt64Ty(llvmContext), ::llvm::Type::getInt64Ty(llvmContext),
                             ::llvm::Type::getInt64Ty(llvmContext)}));
                        break;
                    case DataType::INT64:
                        resTypes.push_back(llvm::StructType::create(
                            llvmContext,
                            {llvm::Type::getInt8PtrTy(llvmContext), llvm::Type::getInt8PtrTy(llvmContext),
                             ::llvm::Type::getInt64Ty(llvmContext), ::llvm::Type::getInt64Ty(llvmContext),
                             ::llvm::Type::getInt64Ty(llvmContext)}));
                        break;
                    case DataType::DBL:
                        resTypes.push_back(llvm::StructType::create(
                            llvmContext,
                            {llvm::Type::getInt8PtrTy(llvmContext), llvm::Type::getInt8PtrTy(llvmContext),
                             ::llvm::Type::getInt64Ty(llvmContext), ::llvm::Type::getInt64Ty(llvmContext),
                             ::llvm::Type::getInt64Ty(llvmContext)}));
                        break;
                    case DataType::BOOL:
                    case DataType::STRING:
                        throw NotImplementedException();
                    case DataType::VOID:
                    case DataType::NUMERIC:
                        break;
                    case DataType::UNKNOWN:
                        throw std::logic_error("");
                    }
                }
            }

            ::llvm::DataLayout layout(llvmModule.get());
            auto resStruct = llvm::StructType::create(llvmContext, resTypes);
            const auto resLayout = layout.getStructLayout(resStruct);
            params.push_back(std::malloc(resLayout->getSizeInBytes()));
            llvm::dbgs() << "Running JIT Program";
            runJIT(config.ObjectFile.empty() ? std::nullopt : std::optional(config.ObjectFile));
            llvm::dbgs() << "Finished Running JIT Program";
            std::shared_ptr<void> basePtr(params.pop_back_val(), ProgramResultDeleter());

            for (size_t i = 0; i < types.size(); ++i)
            {
                switch (types[i])
                {
                case DataType::INT32:
                    res = resolveAndFetchResult<uint32_t>(
                        arities[i], basePtr,
                        std::reinterpret_pointer_cast<char>(basePtr).get() + resLayout->getElementOffset(i), res);
                    break;
                case DataType::INT64:
                    res = resolveAndFetchResult<uint64_t>(
                        arities[i], basePtr,
                        std::reinterpret_pointer_cast<char>(basePtr).get() + resLayout->getElementOffset(i), res);
                    break;
                case DataType::DBL:
                    res = resolveAndFetchResult<double>(
                        arities[i], basePtr,
                        std::reinterpret_pointer_cast<char>(basePtr).get() + resLayout->getElementOffset(i), res);
                    break;
                default:
                    throw std::logic_error("");
                }
            }
        }
        // TODO: separate functions for cleanup in future
        params.clear();
        nparam = 0;
        for (auto elem : toDealloc)
        {
            std::free(elem);
        }
        toDealloc.clear();
        return res;
    }

    void Program::inferTypes()
    {
        TypeInferencePass typeInference(inferer);
        typeInference.inferTypes(*this);
    }

    Program::Program(const std::string &source_path, Config config)
        : func_vars(),
          context(),
          llvmContext(),
          mlirModule(),
          llvmModule(),
          maybeEngine(std::nullopt),
          functions(),
          config{std::move(config)},
          max_in_table_size(0),
          inferer(this)
    {

        if (source_path.empty())
            return;

        auto fst = std::unique_ptr<std::istream>(source_path == "-" ? &std::cin
                                                                    : new std::ifstream(source_path, std::ios::in));

        if (fst->good())
        {
            lexer = std::make_unique<Lexer>(*fst); // read file, decode UTF-8/16/32 format
            lexer->filename = source_path;         // the filename to display with error locations

            ::voila::parser::Parser parser(*lexer, *this);
            if (parser() != 0)
                throw ::voila::ParsingError();
        }
        else
        {
            llvm::errs() << "failed to read from input" << source_path;
        }

        // llvm::DebugFlag = config.debug;
    }

    Program::Program(Config config) : Program("", std::move(config)) {}

    Program::~Program() = default;

    /**
     * @param data pointer to data
     * @param size  number of elements in @link{data} pointer, 0 if pointer to scalar
     * @param type data type compatible to values in @link{data}
     * @return
     */
    Parameter make_param(void *data, size_t size, DataType type) { return {data, size, type}; }
} // namespace voila
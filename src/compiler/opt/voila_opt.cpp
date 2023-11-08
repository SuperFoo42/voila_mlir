#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/IR/MLIRContext.h>                         // for registerMLIR...
#include <mlir/InitAllDialects.h>                        // for registerAllD...
#include <mlir/InitAllPasses.h>                          // for registerAllP...
#include <mlir/Tools/mlir-opt/MlirOptMain.h>             // for MlirOptMain
#include "mlir/Dialects/Voila/IR/VoilaOpsDialect.h.inc"  // for VoilaDialect
#include "mlir/IR/DialectRegistry.h"                     // for DialectRegistry
#include <llvm/Support/CommandLine.h>                    // for opt, desc
#include <charconv>                                      // for from_chars
#include <cstdlib>                                       // for size_t, EXIT...
#include <filesystem>                                    // for is_regular_file
#include <magic_enum.hpp>                                // for enum_cast
#include <optional>                                      // for optional
#include <stdexcept>                                     // for invalid_argu...
#include <string>                                        // for string, oper...
#include <system_error>                                  // for errc
#include <vector>                                        // for vector
#include "Program.hpp"                                   // for Program, mak...
#include "Types.hpp"                                     // for Arity, DataT...
#include "llvm/ADT/SmallVector.h"                        // for SmallVector
#include "llvm/ADT/StringRef.h"                          // for StringRef
#include "mlir/Dialects/Voila/IR/VoilaOpsDialect.h.inc"  // for VoilaDialect
#include "mlir/IR/DialectRegistry.h"
#pragma GCC diagnostic pop

#include "mlir/Dialect/Func/Extensions/AllExtensions.h"

#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

namespace cl = llvm::cl;

namespace {
    struct VType {
        voila::DataType type;
        voila::Arity ar;
    };

    enum Action {
        None,
        DumpAST,
        DumpMLIR,
        DumpMLIRAffine,
        DumpMLIRLLVM,
        DumpLLVMIR,
        RunJIT
    };
}


namespace voila {
    template<>
    Parameter make_param(VType *val) {
        return make_param(nullptr, val->ar.get_size(), val->type);
    }
}

namespace llvm::cl {
    using voila::InputType;

    template<>
    class cl::parser<VType> : public basic_parser<VType> {
      public:
        using parser_data_type = VType;

        explicit parser(Option &O) : basic_parser(O) {}

        bool parse(cl::Option &O, StringRef, const StringRef ArgValue,
                   VType &Val) {
            std::string::size_type n = ArgValue.find(':');
            auto maybeType = magic_enum::enum_cast<voila::DataType>(ArgValue.substr(0, n));
            if (maybeType.has_value()) {
                Val.type = maybeType.value();
            } else {
                O.error("TODO");
                return true;
            }

            if (n != llvm::StringRef::npos) {
                auto as = ArgValue.substr(n + 1);
                size_t result{};

                auto [ptr, ec] {std::from_chars(as.data(), as.data() + as.size(), result)};

                if (ec == std::errc()) {
                    Val.ar = voila::Arity(result);
                } else if (ec == std::errc::invalid_argument) {
                    O.error("That isn't a number.");
                    return true;
                } else if (ec == std::errc::result_out_of_range) {
                    O.error("This number is larger than an size_t.");
                    return true;
                }
            } else {
                Val.ar = voila::Arity(0);
            }

            return false;
        }

        [[nodiscard]] StringRef getValueName() const override { return "Voila Type"; }
    };
}

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input voila file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

static cl::opt<std::string> outputFilename("o", cl::desc("output mlir filename"), cl::init("-"),
                                           cl::value_desc("filename"));

cl::list<VType> paramTypes(cl::Positional, cl::desc("<parameter types in form TYPE:Cardinality>"));

static cl::opt<enum InputType> inputType(
    "x", cl::init(InputType::Voila), cl::desc("Decided the kind of output desired"),
    cl::values(clEnumValN(InputType::Voila, "voila", "load the input file as a voila source.")),
    cl::values(clEnumValN(InputType::MLIR, "mlir",
                          "load the input file as an MLIR file")));

static cl::opt<enum Action> emitAction(
    "emit", cl::desc("Select the kind of output desired"),
    cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    cl::values(clEnumValN(DumpMLIRAffine, "mlir-affine",
                          "output the MLIR dump after affine lowering")),
    cl::values(clEnumValN(DumpMLIRLLVM, "mlir-llvm",
                          "output the MLIR dump after llvm lowering")),
    cl::values(clEnumValN(DumpLLVMIR, "llvm", "output the LLVM IR dump")),
    cl::values(
        clEnumValN(RunJIT, "jit",
                   "JIT the code and run it by invoking the main function")));

static cl::opt<bool> enableOpt("opt", cl::desc("Enable optimizations"));


/*/// Returns a Voila AST resulting from parsing the file or a nullptr on error.
std::unique_ptr<toy::ModuleAST> parseInputFile(llvm::StringRef filename) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(filename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return nullptr;
    }
    auto buffer = fileOrErr.get()->getBuffer();
    LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
    Parser parser(lexer);
    return parser.parseModule();
}

int loadMLIR(mlir::MLIRContext &context,
             mlir::OwningOpRef<mlir::ModuleOp> &module) {
    // Handle '.voila' input to the compiler.
    if (inputType != InputType::MLIR &&
        !llvm::StringRef(inputFilename).endswith(".mlir")) {
        auto moduleAST = parseInputFile(inputFilename);
        if (!moduleAST)
            return 6;
        module = mlirGen(context, *moduleAST);
        return !module ? 1 : 0;
    }

    // Otherwise, the input is '.mlir'.
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }

    // Parse the input mlir.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
    }
    return 0;
}*/

int loadAndProcessMLIR(mlir::MLIRContext &context,
                       mlir::OwningOpRef<mlir::ModuleOp> &module) {
    if (int error = loadMLIR(context, module))
        return error;

    mlir::PassManager pm(module.get()->getName());
    // Apply any generic pass manager command line options and run the pipeline.
    if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
        return 4;

    // Check to see what granularity of MLIR we are compiling to.
    bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
    bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

    if (enableOpt || isLoweringToAffine) {
        // Inline all functions into main and then delete them.
        pm.addPass(mlir::createInlinerPass());

        // Now that there is only one function, we can infer the shapes of each of
        // the operations.
        mlir::OpPassManager &optPM = pm.nest<mlir::voila::FuncOp>();
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::toy::createShapeInferencePass());
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::createCSEPass());
    }

    if (isLoweringToAffine) {
        // Partially lower the toy dialect.
        pm.addPass(mlir::toy::createLowerToAffinePass());

        // Add a few cleanups post lowering.
        mlir::OpPassManager &optPM = pm.nest<mlir::func::FuncOp>();
        optPM.addPass(mlir::createCanonicalizerPass());
        optPM.addPass(mlir::createCSEPass());

        // Add optimizations if enabled.
        if (enableOpt) {
            optPM.addPass(mlir::affine::createLoopFusionPass());
            optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
        }
    }

    if (isLoweringToLLVM) {
        prog.convertToLLVM()
    }

    if (mlir::failed(pm.run(*module)))
        return 4;
    return 0;
}

int dumpAST() {
    if (inputType == InputType::MLIR) {
        llvm::errs() << "Can't dump a Toy AST when the input is MLIR\n";
        return 5;
    }

    auto moduleAST = parseInputFile(inputFilename);
    if (!moduleAST)
        return 1;

    dump(*moduleAST);
    return 0;
}

int dumpLLVMIR(mlir::ModuleOp module) {
    // Register the translation to LLVM IR with the MLIR context.
    mlir::registerBuiltinDialectTranslation(*module->getContext());
    mlir::registerLLVMDialectTranslation(*module->getContext());

    // Convert the module to LLVM IR in a new LLVM IR context.
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return -1;
    }

    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Create target machine and configure the LLVM Module
    auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!tmBuilderOrError) {
        llvm::errs() << "Could not create JITTargetMachineBuilder\n";
        return -1;
    }

    auto tmOrError = tmBuilderOrError->createTargetMachine();
    if (!tmOrError) {
        llvm::errs() << "Could not create TargetMachine\n";
        return -1;
    }
    mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                          tmOrError.get().get());

    /// Optionally run an optimization pipeline over the llvm module.
    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);
    if (auto err = optPipeline(llvmModule.get())) {
        llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
        return -1;
    }
    llvm::errs() << *llvmModule << "\n";
    return 0;
}

int runJit(mlir::ModuleOp module) {
    // Initialize LLVM targets.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    // Register the translation from MLIR to LLVM IR, which must happen before we
    // can JIT-compile.
    mlir::registerBuiltinDialectTranslation(*module->getContext());
    mlir::registerLLVMDialectTranslation(*module->getContext());

    // An optimization pipeline to use within the execution engine.
    auto optPipeline = mlir::makeOptimizingTransformer(
        /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
        /*targetMachine=*/nullptr);

    // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
    // the module.
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = optPipeline;
    auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
    assert(maybeEngine && "failed to construct an execution engine");
    auto &engine = maybeEngine.get();

    // Invoke the JIT-compiled function.
    auto invocationResult = engine->invokePacked("main");
    if (invocationResult) {
        llvm::errs() << "JIT invocation failed\n";
        return -1;
    }

    return 0;
}

int main(int argc, char **argv) {
    // Register any command line options.
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();

    cl::ParseCommandLineOptions(argc, argv, "voila compiler\n");

    if (inputFilename != "-" && !std::filesystem::is_regular_file(std::filesystem::path(inputFilename.data()))) {
        throw std::invalid_argument("invalid input file");
    }

    if (outputFilename != "-" && !std::filesystem::is_regular_file(std::filesystem::path(outputFilename.data()))) {
        throw std::invalid_argument("invalid output file");
    }

    auto prog = voila::Program(inputFilename);

    for (auto param: paramTypes)
        prog << &param;
    prog.inferTypes();


    switch (emitAction) {
    case None:
        llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
        return EXIT_FAILURE;
    case DumpAST:
        prog.to_dot(outputFilename);
        break;
    case DumpMLIR:
        prog.generateMLIR();
        prog.printMLIR(outputFilename);
        break;
    case DumpMLIRAffine:
        prog.generateMLIR();
        prog.generateAffine();
        prog.printMLIR(outputFilename);
        break;
    case DumpMLIRLLVM:
        prog.generateMLIR();
        prog.generateAffine();
        prog.generateLLVM();
        prog.printMLIR(outputFilename);
        break;
    case DumpLLVMIR:
        prog.generateMLIR();
        prog.generateAffine();
        prog.generateLLVM();
        prog.printLLVM(outputFilename);
        break;
    case RunJIT:
        prog.run();
        break;
    }

    return EXIT_SUCCESS;

    // If we aren't dumping the AST, then we are compiling with/to MLIR.
    mlir::DialectRegistry registry;
    mlir::func::registerAllExtensions(registry);

    mlir::MLIRContext context(registry);
    // Load our Dialect in this MLIR Context.
    context.getOrLoadDialect<mlir::voila::VoilaDialect>();
    registerAllDialects(registry);

    mlir::OwningOpRef<mlir::ModuleOp> module;
    if (int error = loadAndProcessMLIR(context, module))
        return error;

    // If we aren't exporting to non-mlir, then we are done.
    bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
    if (isOutputingMLIR) {
        module->dump();
        return EXIT_SUCCESS;
    }

    // Check to see if we are compiling to LLVM IR.
    if (emitAction == Action::DumpLLVMIR)
        return dumpLLVMIR(*module);

    // Otherwise, we must be running the jit.
    if (emitAction == Action::RunJIT)
        return runJit(*module);


    return EXIT_FAILURE;
}
#include "MLIRGenerator.hpp"
#include "ParsingError.hpp"
#include "Program.hpp"
#include "voila_lexer.hpp"
#include "voila_parser.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"

#include "llvm/Support/ToolOutputFile.h"

#include <mlir/Pass/PassManager.h>
#include "mlir/ShapeInferencePass.hpp"
#include "mlir/VoilaDialect.h"
#include "mlir/lowering/VoilaToAffineLoweringPass.hpp"
#pragma GCC diagnostic pop

#include <cstdlib>
#include <cxxopts.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <fstream>

voila::Program parse(const std::string &file)
{
    if (!std::filesystem::is_regular_file(std::filesystem::path(file)))
    {
        throw std::invalid_argument("invalid file");
    }
    voila::Program prog;
    std::ifstream fst(file, std::ios::in);

    if (fst.is_open())
    {
        voila::lexer::Lexer lexer(fst); // read file, decode UTF-8/16/32 format
        lexer.filename = file;          // the filename to display with error locations

        voila::parser::Parser parser(lexer, prog);
        if (parser() != 0)
            throw ParsingError();
    }
    else
    {
        std::cout << fmt::format("failed to open {}", file) << std::endl;
    }

    return prog;
}

int main(int argc, char *argv[])
{
    mlir::registerAllPasses();
    mlir::DialectRegistry registry;
    registry.insert<mlir::voila::VoilaDialect>();
    registry.insert<mlir::StandardOpsDialect>();
    registerAllDialects(registry);
    mlir::registerMLIRContextCLOptions();

    cxxopts::Options options("VOILA compiler", "");

    options.add_options()("h, help", "Show help")(
        "f, file", "File name",
        cxxopts::value<std::string>()) ("a, plot-ast", "Generate dot file of AST",
                                        cxxopts::value<bool>()->default_value(
                                            "false")) ("O,opt", "Enable optimization passes",
                                                       cxxopts::value<bool>()->default_value(
                                                           "true")) ("d, dump-mlir", "Dump intermediate voila",
                                                                     cxxopts::value<bool>()->default_value(
                                                                         "false")) ("l, dump-lowered",
                                                                                    "Dump lowered mlir",
                                                                                    cxxopts::value<bool>()
                                                                                        ->default_value("false"));

    try
    {
        auto cmd = options.parse(argc, argv);

        if (cmd.count("h"))
        {
            std::cout << options.help({"", "Group"}) << std::endl;
            exit(EXIT_SUCCESS);
        }

        const auto file = cmd["f"].as<std::string>();
        if (!std::filesystem::is_regular_file(std::filesystem::path(file)))
        {
            throw std::invalid_argument("invalid file");
        }
        voila::Program prog;
        voila::lexer::Lexer lexer;
        std::ifstream fst(file, std::ios::in);

        if (fst.is_open())
        {
            lexer = voila::lexer::Lexer(fst); // read file, decode UTF-8/16/32 format
            lexer.filename = file;            // the filename to display with error locations

            voila::parser::Parser parser(lexer, prog);
            if (parser() != 0)
                throw ParsingError();
        }
        else
        {
            std::cout << fmt::format("failed to open {}", file) << std::endl;
        }

        if (cmd.count("a"))
        {
            prog.to_dot(cmd["f"].as<std::string>());
        }

        // TODO:
        prog.set_main_args_shape(std::unordered_map<std::string, size_t>(
            {std::pair<std::string, size_t>("x", 1), std::pair<std::string, size_t>("y", 1)}));

        mlir::MLIRContext context;
        // Load our Dialect in this MLIR Context.
        context.getOrLoadDialect<mlir::voila::VoilaDialect>();
        auto module = voila::MLIRGenerator::mlirGen(context, prog);

        if (!module)
            return EXIT_FAILURE;

        if (cmd.count("d"))
        {
            std::error_code ec;
            llvm::raw_fd_ostream os(cmd["f"].as<std::string>() + ".mlir", ec, llvm::sys::fs::OF_None);
            module->print(os);
            os.flush();
        }

        ::mlir::PassManager pm(&context);
        // Apply any generic pass manager command line options and run the pipeline.
        applyPassManagerCLOptions(pm);
         pm.addPass(mlir::createInlinerPass());
        ::mlir::OpPassManager &optPM = pm.nest<mlir::FuncOp>();
        // Now that there is only one function, we can infer the shapes of each of
        // the operations.
        optPM.addPass(voila::mlir::createShapeInferencePass()); // TODO
        //optPM.addPass(mlir::createCanonicalizerPass());
        //optPM.addPass(mlir::createCSEPass());

        // Partially lower the toy dialect with a few cleanups afterwards.
        //optPM.addPass(voila::mlir::createLowerToAffinePass());
        // optPM.addPass(mlir::createCanonicalizerPass());
        // optPM.addPass(mlir::createCSEPass());

        if (cmd.count("O"))
        {
            // optPM.addPass(mlir::createLoopFusionPass());
            // optPM.addPass(mlir::createMemRefDataFlowOptPass());
        }
        if (mlir::failed(pm.run(*module)))
            return EXIT_FAILURE;

        if (cmd.count("l"))
        {
            std::error_code ec;
            llvm::raw_fd_ostream os(cmd["f"].as<std::string>() + ".lowered.mlir", ec, llvm::sys::fs::OF_None);
            module->print(os);
            os.flush();
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cerr << "error parsing options: " << e.what() << std::endl;
        std::cout << options.help({"", "Group"}) << std::endl;
        exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
    // return failed(mlir::MlirOptMain(argc, argv, "Voila optimizer driver\n", registry));
}
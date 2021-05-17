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
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#pragma GCC diagnostic pop
#include "mlir/VoilaDialect.h"

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
        fst.close();
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
        "f,file", "File name", cxxopts::value<std::string>()) ("a, plot-ast", "Generate dot file of AST",
                                                               cxxopts::value<bool>()->default_value("false"));

    try
    {
        auto cmd = options.parse(argc, argv);

        if (cmd.count("h"))
        {
            std::cout << options.help({"", "Group"}) << std::endl;
            exit(EXIT_SUCCESS);
        }

        auto prog = parse(cmd["f"].as<std::string>());
        if (cmd.count("a"))
        {
            prog.to_dot(cmd["f"].as<std::string>());
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cerr << "error parsing options: " << e.what() << std::endl;
        std::cout << options.help({"", "Group"}) << std::endl;
        exit(EXIT_FAILURE);
    }

    return failed(mlir::MlirOptMain(argc, argv, "Voila optimizer driver\n", registry));
}
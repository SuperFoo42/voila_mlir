#include "ParsingError.hpp"
#include "Program.hpp"
#include "voila_lexer.hpp"
#include "MLIRGenerator.hpp"
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
#include "MLIRGenerator.hpp"

#include <cstdlib>
#include <cstdio>
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

    options.add_options()("h, help", "Show help")
        ("f,file", "File name", cxxopts::value<std::string>())
            ("a, plot-ast", "Generate dot file of AST", cxxopts::value<bool>()->default_value("false"))
            ("d, dump-mlir", "Dump intermediate voila", cxxopts::value<bool>()->default_value("false"));

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
            lexer.filename = file;          // the filename to display with error locations

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

        if (cmd.count("d"))
        {
            mlir::MLIRContext context;
            // Load our Dialect in this MLIR Context.
            context.getOrLoadDialect<mlir::voila::VoilaDialect>();
            auto res = voila::MLIRGenerator::mlirGen(context, prog);

            if (!res)
                return EXIT_FAILURE;

            res->dump();
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cerr << "error parsing options: " << e.what() << std::endl;
        std::cout << options.help({"", "Group"}) << std::endl;
        exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
    //return failed(mlir::MlirOptMain(argc, argv, "Voila optimizer driver\n", registry));
}
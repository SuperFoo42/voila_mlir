#include "ParsingError.hpp"
#include "Program.hpp"
#include "voila_lexer.hpp"
#include "voila_parser.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/VoilaDialect.h>
#include <mlir/lowering/VoilaToAffineLoweringPass.hpp>
#pragma GCC diagnostic pop

#include <cstdlib>
#include <cxxopts.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <fstream>
#include <spdlog/spdlog.h>

//TODO: throwing away the lexer/parser leads to deletion of locations and subsequently lookup of invalid memory
std::unique_ptr<::voila::Program> parse(const std::string &file)
{
    if (!std::filesystem::is_regular_file(std::filesystem::path(file)))
    {
        throw std::invalid_argument("invalid file");
    }
    auto prog = std::make_unique<::voila::Program>();
    std::ifstream fst(file, std::ios::in);

    if (fst.is_open())
    {
        ::voila::lexer::Lexer lexer(fst); // read file, decode UTF-8/16/32 format
        lexer.filename = file;            // the filename to display with error locations

        ::voila::parser::Parser parser(lexer, *prog);
        if (parser() != 0)
            throw ::voila::ParsingError();
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

    options.add_options()("h, help", "Show help")("f, file", "File name",cxxopts::value<std::string>())
                        ("a, plot-ast", "Generate dot file of AST", cxxopts::value<bool>()->default_value("false"))
                        ("O,opt", "Enable optimization passes", cxxopts::value<bool>()->default_value("true"))
                        ("o,obj", "Dump object code", cxxopts::value<bool>()->default_value("false"))
                        ("d, dump-mlir", "Dump intermediate voila", cxxopts::value<bool>()->default_value("false"))
                        ("l, dump-lowered","Dump lowered mlir", cxxopts::value<bool>()->default_value("false"))
                        ("j, jit", "jit compile and run code", cxxopts::value<bool>()->default_value("false"))
                        ("v, verbose", "show more info", cxxopts::value<bool>()->default_value("false"));

    try
    {
        auto cmd = options.parse(argc, argv);

        if (cmd.count("h"))
        {
            std::cout << options.help({"", "Group"}) << std::endl;
            exit(EXIT_SUCCESS);
        }

        if (cmd.count("v"))
        {
            spdlog::set_level(spdlog::level::debug);
            ::llvm::DebugFlag = true;
        }
        else
        {
            spdlog::set_level(spdlog::level::warn);
        }

        spdlog::debug("Start parsing input file");
        const auto file = cmd["f"].as<std::string>();
        if (!std::filesystem::is_regular_file(std::filesystem::path(file)))
        {
            throw std::invalid_argument("invalid file");
        }
        auto prog = std::make_unique<::voila::Program>(); //TODO: cmd.count("v")
        ::voila::lexer::Lexer lexer;
        std::ifstream fst(file, std::ios::in);

        if (fst.is_open())
        {
            lexer = ::voila::lexer::Lexer(fst); // read file, decode UTF-8/16/32 format
            lexer.filename = file;              // the filename to display with error locations

            ::voila::parser::Parser parser(lexer, *prog);
            if (parser() != 0)
                throw ::voila::ParsingError();
        }
        else
        {
            std::cout << fmt::format("failed to open {}", file) << std::endl;
        }

        if (cmd.count("a"))
        {
            prog->to_dot(cmd["f"].as<std::string>());
        }
        spdlog::debug("Finished parsing input file");

        // TODO:
        prog->set_main_args_type(std::unordered_map<std::string, voila::DataType>(
            {std::pair<std::string, voila::DataType>("x", voila::DataType::INT64), std::pair<std::string, voila::DataType>("y", voila::DataType::INT64)}));
        prog->set_main_args_shape(std::unordered_map<std::string, size_t>(
            {std::pair<std::string, size_t>("x", 100), std::pair<std::string, size_t>("y", 100)}));
        spdlog::debug("Start mlir generation");
        //generate mlir
        prog->generateMLIR();

        if (cmd.count("d"))
        {
            prog->printMLIR(cmd["f"].as<std::string>());
        }
        spdlog::debug("Finished mlir generation");
        spdlog::debug("Start mlir lowering");
        //lower mlir
        prog->lowerMLIR(cmd.count("O"));
        spdlog::debug("Finished mlir lowering");
        spdlog::debug("Start mlir to llvm conversion");
        //lower to llvm
        prog->convertToLLVM(cmd.count("O"));

        if (cmd.count("l"))
        {
            prog->printLLVM(cmd["f"].as<std::string>());
        }
        spdlog::debug("Finished mlir to llvm conversion");
        //run in jit
        if (cmd.count("j"))
        {
            spdlog::debug("Running program");
            if (cmd.count("o"))
                prog->runJIT(cmd.count("O"), cmd["f"].as<std::string>());
            else
                prog->runJIT(cmd.count("O"), std::nullopt);
            spdlog::debug("Finished Running program");
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
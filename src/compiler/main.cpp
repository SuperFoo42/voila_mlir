#include "ParsingError.hpp"
#include "Program.hpp"

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
#include <fstream>
#include <spdlog/spdlog.h>

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
        cxxopts::value<
            std::
                string>()) ("a, plot-ast", "Generate dot file of AST",
                            cxxopts::value<bool>()->default_value(
                                "false")) ("O,opt", "Enable optimization passes",
                                           cxxopts::value<bool>()->default_value(
                                               "true")) ("o,obj", "Dump object code",
                                                         cxxopts::value<bool>()->default_value(
                                                             "false")) ("d, dump-mlir", "Dump intermediate voila",
                                                                        cxxopts::value<bool>()->default_value(
                                                                            "false")) ("l, dump-lowered",
                                                                                       "Dump lowered mlir",
                                                                                       cxxopts::value<bool>()->default_value(
                                                                                           "false")) ("j, jit",
                                                                                                      "jit compile and "
                                                                                                      "run code",
                                                                                                      cxxopts::value<
                                                                                                          bool>()
                                                                                                          ->default_value(
                                                                                                              "fals"
                                                                                                              "e")) ("v"
                                                                                                                     ","
                                                                                                                     " "
                                                                                                                     "v"
                                                                                                                     "e"
                                                                                                                     "r"
                                                                                                                     "b"
                                                                                                                     "o"
                                                                                                                     "s"
                                                                                                                     "e",
                                                                                                                     "s"
                                                                                                                     "h"
                                                                                                                     "o"
                                                                                                                     "w"
                                                                                                                     " "
                                                                                                                     "m"
                                                                                                                     "o"
                                                                                                                     "r"
                                                                                                                     "e"
                                                                                                                     " "
                                                                                                                     "i"
                                                                                                                     "n"
                                                                                                                     "f"
                                                                                                                     "o",
                                                                                                                     cxxopts::value<
                                                                                                                         bool>()
                                                                                                                         ->default_value(
                                                                                                                             "false"));

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
        auto prog = ::voila::Program(file, cmd.count("v"), cmd.count("O"));

        if (cmd.count("a"))
        {
            prog.to_dot(file);
        }
        spdlog::debug("Finished parsing input file");

        // alloc dummy data to pass to program args
        auto *arg = static_cast<uint64_t *>(std::malloc(sizeof(uint64_t) * 1000));
        std::fill_n(arg, 1000, 123);
        auto *arg2 = static_cast<uint64_t *>(std::malloc(sizeof(uint64_t) * 100));
        std::fill_n(arg2, 100, 123);
        prog << ::voila::make_param(arg, 100, voila::DataType::INT64);
        prog << ::voila::make_param(arg2, 100, voila::DataType::INT64);

        // run in jit
        auto res = prog();
        for (auto elem : *(*std::get<std::unique_ptr<StridedMemRefType<uint64_t, 1> *>>(res)))
            std::cout << elem << std::endl;
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cerr << "error parsing options: " << e.what() << std::endl;
        std::cout << options.help({"", "Group"}) << std::endl;
        exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}
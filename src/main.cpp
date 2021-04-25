#include "ParsingError.hpp"
#include "voila_lexer.hpp"
#include "voila_parser.hpp"

#include <ast/DotVisualizer.hpp>
#include <cstdlib>
#include <cxxopts.hpp>
#include <filesystem>
#include <fmt/core.h>
#include <fstream>
std::vector<Fun> parse(const std::string &file)
{
    if (!std::filesystem::is_regular_file(std::filesystem::path(file)))
    {
        throw std::invalid_argument("invalid file");
    }
    std::vector<Fun> funcs;
    std::ifstream fst(file, std::ios::in);

    if (fst.is_open())
    {
        voila::lexer::Lexer lexer(fst); // read file, decode UTF-8/16/32 format
        lexer.filename = file;        // the filename to display with error locations

        voila::parser::Parser parser(lexer, funcs);
        if (parser() != 0)
            throw ParsingError();
        fst.close();
    }
    else
    {
        std::cout << fmt::format("failed to open {}", file) << std::endl;
    }

    return funcs;
}

void asts_to_dot(const std::string &outFileName, std::vector<Fun> &funcs)
{
    for (auto &func : funcs)
    {
        DotVisualizer vis(func);
        std::ofstream out(outFileName+"."+func.name+".dot", std::ios::out);
        out << vis;
        out.close();
    }
}
int main(int argc, char *argv[])
{
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

        auto funcs = parse(cmd["f"].as<std::string>());
        if (cmd.count("a"))
        {
            asts_to_dot(cmd["f"].as<std::string>(), funcs);
        }
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cerr << "error parsing options: " << e.what() << std::endl;
        std::cout << options.help({"", "Group"}) << std::endl;
        exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}
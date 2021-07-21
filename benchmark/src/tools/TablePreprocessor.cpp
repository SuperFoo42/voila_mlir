#include "ColumnTypes.hpp"
#include "TableReader.hpp"
#include "Tables.hpp"
#include <cxxopts.hpp>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char **argv)
{
    cxxopts::Options options("VOILA benchmark preprocessing tool",
                             "This tool prepares csv files for use in the benchmarks. Therefore, it dictionary "
                             "compresses string columns and converts date columns to int");
    options.add_options()("h, help", "Show help")(
        "f, file", "File name",
        cxxopts::value<std::string>()) ("d, delim", "Delimiter used in csv file",
                                        cxxopts::value<char>()->default_value(
                                            "|")) ("t, types", "Column types in csv file",
                                                   cxxopts::value<std::vector<
                                                       ColumnTypes>>()) ("o, output",
                                                                         "Filename to write preprocessed table",
                                                                         cxxopts::value<std::string>());

    try
    {
        auto cmd = options.parse(argc, argv);
        if (cmd.count("h"))
        {
            std::cout << options.help({"", "Group"}) << std::endl;
            exit(EXIT_SUCCESS);
        }

        TableReader<CompressedTable> reader(cmd["f"].as<std::string>(), cmd["t"].as<std::vector<ColumnTypes>>(),
                                            cmd["d"].as<char>());
        auto compressedTable = reader.getTable();
        Table::writeTable(cmd["o"].as<std::string>(), compressedTable);
    }
    catch (const cxxopts::OptionException &e)
    {
        std::cerr << "error parsing options: " << e.what() << std::endl;
        std::cout << options.help({"", "Group"}) << std::endl;
        exit(EXIT_FAILURE);
    }
}

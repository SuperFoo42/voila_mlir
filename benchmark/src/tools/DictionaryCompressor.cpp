#include "DictionaryCompressor.hpp"

#include <string>
#include <unordered_map>
#include <vector>

std::pair<std::vector<int32_t>, std::unordered_map<std::string, int32_t>> DictionaryCompressor::compress()
{
    std::unordered_map<std::string, int32_t> dictionary;
    {
        std::set<std::string> colValues(column.begin(), column.end());

        for (auto &elem : colValues)
            dictionary.emplace(elem, static_cast<int64_t>(dictionary.size())); //TODO: sth. is wrong with this
    }
    std::vector<int32_t> compressedCol;
    compressedCol.reserve(column.size());

    for (auto &&elem : column)
    {
        compressedCol.push_back(dictionary.at(elem));
    }
    return std::make_pair(compressedCol, dictionary);
}
DictionaryCompressor::DictionaryCompressor(std::vector<std::string> &&column) : column(column) {}

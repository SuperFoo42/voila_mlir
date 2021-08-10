#pragma once
#include "DictionaryCompressor.hpp"

#include <set>
#include <string>
#include <unordered_map>
#include <vector>
class DictionaryCompressor
{
    std::vector<std::string> column;

  public:
    explicit DictionaryCompressor(std::vector<std::string> &&column);
    std::pair<std::vector<int32_t>, std::unordered_map<std::string, int32_t>> compress();
};
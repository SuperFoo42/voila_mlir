#pragma once
#include "magic_enum.hpp"

#include <algorithm>
#include <iostream>
#include <string>

enum class ColumnTypes
{
    INT,
    STRING,
    DATE,
    DECIMAL
};

inline std::istream &operator>>(std::istream &val, ColumnTypes &tbl)
{
    std::string input;
    val >> input;
    std::transform(input.begin(), input.end(), input.begin(), [](unsigned char c) { return std::toupper(c); });

    auto type = magic_enum::enum_cast<ColumnTypes>(input);
    if (type.has_value())
    {
        tbl = type.value();
    }
    else
    {
        throw std::logic_error("Type not parseable");
    }
    return val;
}
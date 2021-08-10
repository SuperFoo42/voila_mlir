#pragma once

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

class DateReformatter
{
    std::vector<std::string> column;

  public:
    explicit DateReformatter(std::vector<std::string> &&column);

    std::vector<int32_t> reformat();
};
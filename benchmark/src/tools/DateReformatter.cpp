#include "DateReformatter.hpp"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
static int64_t parseDate(std::string &date)
{
    tm t = {};
    std::istringstream ss(date);
    //ss.imbue(std::locale(date));
    ss >> std::get_time(&t, "%Y-%m-%d");

    return 10000 * t.tm_year + 100 * t.tm_mon + t.tm_mday;
}
std::vector<int64_t> DateReformatter::reformat()
{
    std::vector<int64_t> res;

    for (auto &elem : column)
    {
        res.push_back(parseDate(elem));
    }

    return res;
}
DateReformatter::DateReformatter(std::vector<std::string> &&column) : column(column) {}

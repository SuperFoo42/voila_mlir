#include "DateReformatter.hpp"

#include <iomanip>
#include <string>
#include <vector>
static int32_t parseDate(std::string &date)
{
    tm t = {};
    if (date == "NULL")
        return 0;
    std::istringstream ss(date);
    // ss.imbue(std::locale(date));
    ss >> std::get_time(&t, "%Y-%m-%d");

    return 10000 * (1900 + t.tm_year) + 100 * t.tm_mon + t.tm_mday;
}

std::vector<int32_t> DateReformatter::reformat()
{
    std::vector<int32_t> res;

    for (auto &elem : column)
    {
        res.push_back(parseDate(elem));
    }

    return res;
}

DateReformatter::DateReformatter(std::vector<std::string> &&column) : column(column) {}

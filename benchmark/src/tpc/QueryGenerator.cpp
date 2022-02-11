#include "QueryGenerator.hpp"

#include "BenchmarkState.hpp"


int32_t QueryGenerator::getQ1Date()
{
    static size_t cur_idx = 0;
    auto res = q1_dates[cur_idx++];
    cur_idx %= q1_dates.size();
    return res;
}
int32_t QueryGenerator::getQ3CompressedSegment(BenchmarkState &state)
{
    return state.getCustomerOrderLineitem().getDictionary(/*customer_cols::C_MKTSEGMENT*/6).at(getQ3Segment());
}

std::string QueryGenerator::getQ3Segment()
{
    static size_t cur_idx = 0;
    auto res = q3_segments[cur_idx++];
    cur_idx %= q3_segments.size();
    return res;
}
int32_t QueryGenerator::getQ3Date()
{
    static size_t cur_idx = 0;
    auto res = q3_dates[cur_idx++];
    cur_idx %= q3_dates.size();
    return res;
}
int32_t QueryGenerator::getQ6Date()
{
    static size_t cur_idx = 0;
    auto res = q6_dates[cur_idx++];
    cur_idx %= q6_dates.size();
    return res;
}
double QueryGenerator::getQ6Discount()
{
    static size_t cur_idx = 0;
    auto res = q6_discounts[cur_idx++];
    cur_idx %= q6_discounts.size();
    return res;
}
int32_t QueryGenerator::getQ6Quantity()
{
    static size_t cur_idx = 0;
    auto res = q6_quantities[cur_idx++];
    cur_idx %= q6_quantities.size();
    return res;
}
std::string QueryGenerator::getQ9Color()
{
    static size_t cur_idx = 0;
    auto res = q9_colors[cur_idx++];
    cur_idx %= q9_colors.size();
    return res;
}
std::vector<int32_t> QueryGenerator::getQ9CompressedColor(BenchmarkState &state)
{
    const auto color = getQ9Color();
    const auto &dict = state.getPartSupplierLineitemPartsuppOrdersNation().getDictionary(/*part_cols::P_NAME*/1);
    std::vector<int32_t> colorSet;
    for (auto &elem : dict)
    {
        if (elem.first.find(color) != std::string::npos)
        {
            colorSet.push_back(elem.second);
        }
    }

    return colorSet;
}
int32_t QueryGenerator::getQ18Quantity()
{
    static size_t cur_idx = 0;
    auto res = q18_quantities[cur_idx++];
    cur_idx %= q18_quantities.size();
    return res;
}
QueryGenerator::QueryGenerator(std::mt19937::result_type seed, int iterations) :
    q1_dates{}, q3_dates{}, q6_dates{}, q6_discounts{}, q6_quantities{}, q3_segments{}, q9_colors{}, q18_quantities{}
{
    std::mt19937 gen(seed);
    // q1
    std::uniform_int_distribution<unsigned int> dist(0, Q1_DATES.size() - 1);
    std::generate_n(std::back_inserter(q1_dates), iterations, [&gen, &dist]() { return Q1_DATES[dist(gen)]; });

    // q3
    dist = std::uniform_int_distribution<unsigned int>(0, SEGMENTS.size() - 1);
    std::generate_n(std::back_inserter(q3_segments), iterations, [&gen, &dist]() { return SEGMENTS[dist(gen)]; });

    dist = std::uniform_int_distribution<unsigned int>(0, Q3_DATES.size() - 1);
    std::generate_n(std::back_inserter(q3_dates), iterations, [&gen, &dist]() { return Q3_DATES[dist(gen)]; });

    dist = std::uniform_int_distribution<unsigned int>(0, Q6_DATES.size() - 1);
    std::generate_n(std::back_inserter(q6_dates), iterations, [&gen, &dist]() { return Q6_DATES[dist(gen)]; });

    dist = std::uniform_int_distribution<unsigned int>(0, DISCOUNTS.size() - 1);
    std::generate_n(std::back_inserter(q6_discounts), iterations, [&gen, &dist]() { return DISCOUNTS[dist(gen)]; });

    dist = std::uniform_int_distribution<unsigned int>(24, 25);
    std::generate_n(std::back_inserter(q6_quantities), iterations, [&gen, &dist]() { return dist(gen); });

    dist = std::uniform_int_distribution<unsigned int>(0, COLORS.size() - 1);
    std::generate_n(std::back_inserter(q9_colors), iterations, [&gen, &dist]() { return COLORS[dist(gen)]; });

    dist = std::uniform_int_distribution<unsigned int>(312, 315);
    std::generate_n(std::back_inserter(q18_quantities), iterations, [&gen, &dist]() { return dist(gen); });
}

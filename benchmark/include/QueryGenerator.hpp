#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>


class BenchmarkState;

class QueryGenerator
{
    constexpr static auto Q1_DATES = std::to_array(
        {19981002, 19981001, 19980930, 19980929, 19980928, 19980927, 19980926, 19980925, 19980924, 19980923, 19980922,
         19980921, 19980920, 19980919, 19980918, 19980917, 19980916, 19980915, 19980914, 19980913, 19980912, 19980911,
         19980910, 19980909, 19980908, 19980907, 19980906, 19980905, 19980904, 19980903, 19980902, 19980901, 19980831,
         19980830, 19980829, 19980828, 19980827, 19980826, 19980825, 19980824, 19980823, 19980822, 19980821, 19980820,
         19980819, 19980818, 19980817, 19980816, 19980815, 19980814, 19980813, 19980812, 19980811, 19980810, 19980809,
         19980808, 19980807, 19980806, 19980805, 19980804, 19980803});

    constexpr static auto Q3_DATES = std::to_array(
        {19950301, 19950302, 19950303, 19950304, 19950305, 19950306, 19950307, 19950308, 19950309, 19950310, 19950311,
         19950312, 19950313, 19950314, 19950315, 19950316, 19950317, 19950318, 19950319, 19950320, 19950321, 19950322,
         19950323, 19950324, 19950325, 19950326, 19950327, 19950328, 19950329, 19950330, 19950331});

    constexpr static auto Q6_DATES = std::to_array({19930101, 19930101, 19950101, 19960101, 19970101});

    constexpr static auto DISCOUNTS = std::to_array({0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09});

    constexpr static auto SEGMENTS = std::to_array({"AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"});

    constexpr static auto COLORS = std::to_array(
        {"almond",    "antique",    "aquamarine", "azure",     "beige",     "bisque",     "black",     "blanched",
         "blue",      "blush",      "brown",      "burlywood", "burnished", "chartreuse", "chiffon",   "chocolate",
         "coral",     "cornflower", "cornsilk",   "cream",     "cyan",      "dark",       "deep",      "dim",
         "dodger",    "drab",       "firebrick",  "floral",    "forest",    "frosted",    "gainsboro", "ghost",
         "goldenrod", "green",      "grey",       "honeydew",  "hot",       "indian",     "ivory",     "khaki",
         "lace",      "lavender",   "lawn",       "lemon",     "light",     "lime",       "linen",     "magenta",
         "maroon",    "medium",     "metallic",   "midnight",  "mint",      "misty",      "moccasin",  "navajo",
         "navy",      "olive",      "orange",     "orchid",    "pale",      "papaya",     "peach",     "peru",
         "pink",      "plum",       "powder",     "puff",      "purple",    "red",        "rose",      "rosy",
         "royal",     "saddle",     "salmon",     "sandy",     "seashell",  "sienna",     "sky",       "slate",
         "smoke",     "snow",       "spring",     "steel",     "tan",       "thistle",    "tomato",    "turquoise",
         "violet",    "wheat",      "white",      "yellow"});

    std::vector<int32_t> q1_dates;
    std::vector<int32_t> q3_dates;
    std::vector<int32_t> q6_dates;
    std::vector<double> q6_discounts;
    std::vector<int32_t> q6_quantities;
    std::vector<std::string> q3_segments;
    std::vector<std::string> q9_colors;
    std::vector<int32_t> q18_quantities;

  public:
    explicit QueryGenerator(std::mt19937::result_type seed = std::mt19937::default_seed, int iterations = 1);

    // substitution parameter generators
    int32_t getQ1Date();

    int32_t getQ3CompressedSegment(BenchmarkState &state);

    std::string getQ3Segment();

    int32_t getQ3Date();

    int32_t getQ6Date();

    double getQ6Discount();

    int32_t getQ6Quantity();

    std::string getQ9Color();

    std::vector<int32_t> getQ9CompressedColor(BenchmarkState &state);

    int32_t getQ18Quantity();
};

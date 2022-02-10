#include "Config.hpp"
#include "Program.hpp"
#include "test_defs.hpp.inc"

#include <gtest/gtest.h>
using namespace voila;

TEST(ProgramTest, MakeParam)
{
    //::voila::make_param(arg);
}

TEST(AddTests, TensorTensorTest)
{
    Config config;

    config.debug().optimize(false);

    const auto file = VOILA_TEST_SOURCES_PATH "/simple_add.voila";
    constexpr size_t TENSOR_SIZE = 100;
    constexpr int64_t TENSOR_VALS = 123;
    constexpr int64_t TENSOR_SUM = TENSOR_VALS + TENSOR_VALS;
    Program prog(file, config);
    // alloc dummy data to pass to program args
    std::vector<int64_t> arg(TENSOR_SIZE, TENSOR_VALS);
    std::vector<int64_t> arg2(TENSOR_SIZE, TENSOR_VALS);
    prog << ::voila::make_param(arg);
    prog << ::voila::make_param(arg2);

    // run in jit
    auto res = std::get<strided_memref_ptr<uint64_t, 1>>(prog()[0]);

    ASSERT_EQ(res->sizes[0], TENSOR_SIZE);
    for (auto elem : *res)
    {
        ASSERT_EQ(elem, TENSOR_SUM);
    }
}

TEST(PredicationTests, AddTest)
{
    Config config;

    config.debug().optimize(false);

    const auto file = VOILA_TEST_SOURCES_PATH "/predicated_add.voila";
    constexpr size_t TENSOR_SIZE = 100;
    std::vector<int32_t> vals(TENSOR_SIZE);
    std::iota(vals.begin(), vals.end(),0);
    Program prog(file, config);
    // alloc dummy data to pass to program args
    prog << ::voila::make_param(vals);
    prog << ::voila::make_param(vals);

    // run in jit
    auto res = std::get<strided_memref_ptr<uint32_t, 1>>(prog()[0]);

    ASSERT_EQ(res->sizes[0], TENSOR_SIZE/2);
    for (auto elem : *res)
    {
        std::cout << elem << "\n";
        //ASSERT_EQ(elem, );
    }
}
/*
TEST(AddTests, TensorScalarTest)
{
    Config config;

    config.debug = true;
    config.optimize = true;

    const auto file = VOILA_TEST_SOURCES_PATH "/simple_add.voila";
    constexpr size_t TENSOR_SIZE = 100;
    constexpr uint64_t TENSOR_VALS = 123;
    constexpr uint64_t TENSOR_SUM = TENSOR_VALS + TENSOR_VALS;
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    std::fill_n(arg.get(), TENSOR_SIZE, TENSOR_VALS);
    auto arg2 = std::unique_ptr<uint64_t>(new uint64_t);
    *arg2 = TENSOR_VALS;
    prog << ::voila::make_param(arg.get(), TENSOR_SIZE, voila::DataType::INT64);
    prog << ::voila::make_param(arg2.get(), 0, voila::DataType::INT64);

    // run in jit
    auto res = std::get<strided_memref_ptr<uint64_t, 1>>(prog()[0]);

    ASSERT_EQ(res->sizes[0], TENSOR_SIZE);
    for (auto elem : *res)
        ASSERT_EQ(elem, TENSOR_SUM);
}

TEST(AddTests, ScalarTensorTest)
{
    FAIL();
}

TEST(AddTests, ScalarScalarTest)
{
    FAIL();
}*/

TEST(SubTests, TensorTensorTest)
{
    Config config;

    config.debug().optimize();

    const auto file = VOILA_TEST_SOURCES_PATH "/simple_sub.voila";
    constexpr size_t TENSOR_SIZE = 100;
    constexpr uint64_t TENSOR_VALS = 123;
    constexpr uint64_t TENSOR_SUB = TENSOR_VALS - TENSOR_VALS;
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    std::fill_n(arg.get(), TENSOR_SIZE, TENSOR_VALS);
    auto arg2 = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    std::fill_n(arg2.get(), TENSOR_SIZE, TENSOR_VALS);
    prog << ::voila::make_param(arg.get(), TENSOR_SIZE);
    prog << ::voila::make_param(arg2.get(), TENSOR_SIZE);

    // run in jit
    auto res = std::get<strided_memref_ptr<uint64_t, 1>>(prog()[0]);

    ASSERT_EQ(res->sizes[0], TENSOR_SIZE);
    for (auto elem : *res)
        ASSERT_EQ(elem, TENSOR_SUB);
}
/*
TEST(SubTests, TensorScalarTest)
{
    FAIL();
}

TEST(SubTests, ScalarTensorTest)
{
    FAIL();
}

TEST(SubTests, ScalarScalarTest)
{
    FAIL();
}*/

TEST(HashTableTests, ScalarHash)
{
    Config config;

    config.debug().optimize(false);

    const auto file = VOILA_TEST_SOURCES_PATH "/simple_hash.voila";
    constexpr size_t TENSOR_SIZE = 100;
    constexpr int64_t TENSOR_VALS = 123;
    constexpr uint64_t HASH = 7668608003591710536;
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<int64_t[]>(new int64_t[TENSOR_SIZE]);
    std::fill_n(arg.get(), TENSOR_SIZE, TENSOR_VALS);
    prog << ::voila::make_param(arg.get(), TENSOR_SIZE);

    // run in jit
    auto res = std::get<strided_memref_ptr<uint64_t, 1>>(prog()[0]);

    ASSERT_EQ(res->sizes[0], TENSOR_SIZE);

    for (size_t i = 0; i < TENSOR_SIZE; ++i)
    {
        ASSERT_EQ(res->operator[](i), HASH);
    }
}

TEST(HashTableTests, VariadicHash)
{
    Config config;

    config.debug().optimize();
    constexpr auto file = VOILA_TEST_SOURCES_PATH "/complex_hash.voila";
    constexpr size_t TENSOR_SIZE = 100;
    constexpr uint64_t TENSOR_VALS1 = 123;
    constexpr uint64_t TENSOR_VALS2 = 246;
    constexpr uint64_t HASH = 10011092986887397633ULL;
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    auto arg2 = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    std::fill_n(arg.get(), TENSOR_SIZE, TENSOR_VALS1);
    std::fill_n(arg2.get(), TENSOR_SIZE, TENSOR_VALS2);
    prog << ::voila::make_param(arg.get(), TENSOR_SIZE);
    prog << ::voila::make_param(arg2.get(), TENSOR_SIZE);

    // run in jit
    auto res = std::get<strided_memref_ptr<uint64_t, 1>>(prog()[0]);

    ASSERT_EQ(res->sizes[0], TENSOR_SIZE);

    for (size_t i = 0; i < TENSOR_SIZE; ++i)
    {
        ASSERT_EQ(res->operator[](i), HASH);
    }
}

TEST(HashTableTests, ScalarInsert)
{
    Config config;

    config.debug().optimize(false);
    const auto file = VOILA_TEST_SOURCES_PATH "/simple_insert.voila";
    constexpr size_t TENSOR_SIZE = 100;
    constexpr size_t NEXTPOW = 128;
    constexpr int64_t TENSOR_VALS = 123;
    constexpr int64_t INVALID = 18446744073709551615ULL;
    constexpr auto ref = std::to_array<int64_t>(
        {INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         123,     INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID});
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<int64_t[]>(new int64_t[TENSOR_SIZE]);
    std::fill_n(arg.get(), TENSOR_SIZE, TENSOR_VALS);
    prog << ::voila::make_param(arg.get(), TENSOR_SIZE);

    // run in jit
    auto res = std::get<strided_memref_ptr<uint64_t, 1>>(prog()[0]);

    ASSERT_EQ(res->sizes[0], NEXTPOW);

    for (size_t i = 0; i < TENSOR_SIZE; ++i)
    {
        ASSERT_EQ(res->operator[](i), ref[i]);
    }
}

// TODO
TEST(HashTableTests, ComplexInsert)
{
    Config config;

    config.debug().optimize();
    constexpr auto file = VOILA_TEST_SOURCES_PATH "/complex_insert.voila";
    constexpr size_t TENSOR_SIZE = 100;
    constexpr uint64_t TENSOR_VALS1 = 123;
    constexpr uint64_t TENSOR_VALS2 = 246;
    constexpr size_t NEXTPOW = 128;
    constexpr auto INVALID = 18446744073709551615ULL;
    constexpr auto ref1 = std::to_array<uint64_t>(
        {INVALID, 123,     INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID});

    constexpr auto ref2 = std::to_array<uint64_t>(
        {INVALID, 246,     INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID,
         INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID, INVALID});
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    auto arg2 = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    std::fill_n(arg.get(), TENSOR_SIZE, TENSOR_VALS1);
    std::fill_n(arg2.get(), TENSOR_SIZE, TENSOR_VALS2);
    prog << ::voila::make_param(arg.get(), TENSOR_SIZE);
    prog << ::voila::make_param(arg2.get(), TENSOR_SIZE);

    // run in jit
    auto res = prog();
    auto t1 = std::get<strided_memref_ptr<uint64_t, 1>>(res.at(0));
    auto t2 = std::get<strided_memref_ptr<uint64_t, 1>>(res.at(1));

    ASSERT_EQ(t1->sizes[0], NEXTPOW);
    ASSERT_EQ(t2->sizes[0], NEXTPOW);

    for (size_t i = 0; i < TENSOR_SIZE; ++i)
    {
        ASSERT_EQ(t1->operator[](i), ref1[i]);
        ASSERT_EQ(t2->operator[](i), ref2[i]);
    }
}

TEST(HashTableTests, SimpleLookup)
{
    Config config;
    //::llvm::DebugFlag = true;
    config.debug().optimize(false);
    const auto file = VOILA_TEST_SOURCES_PATH "/simple_lookup.voila";
    constexpr size_t TENSOR_SIZE = 100;
    constexpr int64_t TENSOR_VALS = 123;
    constexpr size_t VALUE_POS = 72;
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<int64_t[]>(new int64_t[TENSOR_SIZE]);
    std::fill_n(arg.get(), TENSOR_SIZE, TENSOR_VALS);
    prog << ::voila::make_param(arg.get(), TENSOR_SIZE);

    // run in jit
    auto res = std::get<strided_memref_ptr<uint64_t, 1>>(prog()[0]);

    ASSERT_EQ(res->sizes[0], TENSOR_SIZE);

    for (size_t i = 0; i < TENSOR_SIZE; ++i)
    {
        ASSERT_EQ(res->operator[](i), VALUE_POS);
    }
}

TEST(HashTableTests, ComplexLookup)
{
    Config config;

    config.debug().optimize();
    constexpr auto file = VOILA_TEST_SOURCES_PATH "/complex_lookup.voila";
    constexpr size_t TENSOR_SIZE = 100;
    constexpr uint64_t TENSOR_VALS1 = 123;
    constexpr uint64_t TENSOR_VALS2 = 246;
    constexpr size_t VALUE_POS = 1;
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    auto arg2 = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    std::fill_n(arg.get(), TENSOR_SIZE, TENSOR_VALS1);
    std::fill_n(arg2.get(), TENSOR_SIZE, TENSOR_VALS2);
    prog << ::voila::make_param(arg.get(), TENSOR_SIZE);
    prog << ::voila::make_param(arg2.get(), TENSOR_SIZE);

    // run in jit
    auto res = std::get<strided_memref_ptr<uint64_t, 1>>(prog()[0]);

    ASSERT_EQ(res->sizes[0], TENSOR_SIZE);

    for (size_t i = 0; i < TENSOR_SIZE; ++i)
    {
        ASSERT_EQ(res->operator[](i), VALUE_POS);
    }
}

TEST(AggregateTests, SumTest)
{
    Config config;

    config.debug().async_parallel(false).gpu_parallel();

    const auto file = VOILA_TEST_SOURCES_PATH "/simple_sum.voila";
    constexpr size_t TENSOR_SIZE = 100;
    constexpr int64_t TENSOR_VALS = 123;
    constexpr uint64_t TENSOR_SUM = TENSOR_SIZE * TENSOR_VALS;
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<int64_t[]>(new int64_t[TENSOR_SIZE]);
    std::fill_n(arg.get(), TENSOR_SIZE, TENSOR_VALS);
    prog << ::voila::make_param(arg.get(), TENSOR_SIZE);

    // run in jit
    auto res = std::get<uint64_t>(prog()[0]);

    ASSERT_EQ(res, TENSOR_SUM);
}

TEST(AggregateTests, MinTest)
{
    Config config;

    config.debug();
    const auto file = VOILA_TEST_SOURCES_PATH "/simple_min.voila";
    constexpr auto TENSOR_VALS = std::to_array<uint64_t>(
        {441, 965, 381, 125, 626, 162, 930, 213, 969, 866, 235, 571, 822, 469, 350, 73,  150, 494, 629, 236,
         15,  91,  843, 391, 771, 972, 759, 551, 388, 620, 651, 854, 810, 878, 737, 719, 331, 686, 532, 707,
         309, 540, 378, 872, 400, 8,   287, 38,  65,  452, 631, 337, 393, 502, 261, 917, 57,  410, 667, 561,
         431, 960, 64,  358, 488, 366, 820, 849, 529, 621, 890, 268, 230, 528, 87,  20,  117, 258, 794, 644,
         893, 565, 256, 906, 658, 557, 228, 176, 284, 159, 796, 18,  964, 635, 26,  105, 633, 832, 419, 369});
    constexpr auto TENSOR_MIN = *std::min_element(TENSOR_VALS.begin(), TENSOR_VALS.end());
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_VALS.size()]);
    std::copy(TENSOR_VALS.begin(), TENSOR_VALS.end(), arg.get());
    prog << ::voila::make_param(arg.get(), TENSOR_VALS.size());

    // run in jit
    auto res = std::get<uint64_t>(prog()[0]);

    ASSERT_EQ(res, TENSOR_MIN);
}

TEST(AggregateTests, MaxTest)
{
    Config config;

    config.debug();

    const auto file = VOILA_TEST_SOURCES_PATH "/simple_max.voila";
    constexpr auto TENSOR_VALS = std::to_array<uint64_t>(
        {441, 965, 381, 125, 626, 162, 930, 213, 969, 866, 235, 571, 822, 469, 350, 73,  150, 494, 629, 236,
         15,  91,  843, 391, 771, 972, 759, 551, 388, 620, 651, 854, 810, 878, 737, 719, 331, 686, 532, 707,
         309, 540, 378, 872, 400, 8,   287, 38,  65,  452, 631, 337, 393, 502, 261, 917, 57,  410, 667, 561,
         431, 960, 64,  358, 488, 366, 820, 849, 529, 621, 890, 268, 230, 528, 87,  20,  117, 258, 794, 644,
         893, 565, 256, 906, 658, 557, 228, 176, 284, 159, 796, 18,  964, 635, 26,  105, 633, 832, 419, 369});
    constexpr auto TENSOR_MAX = *std::max_element(TENSOR_VALS.begin(), TENSOR_VALS.end());
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_VALS.size()]);
    std::copy(TENSOR_VALS.begin(), TENSOR_VALS.end(), arg.get());
    prog << ::voila::make_param(arg.get(), TENSOR_VALS.size());

    // run in jit
    auto res = std::get<uint64_t>(prog()[0]);

    ASSERT_EQ(res, TENSOR_MAX);
}

TEST(AggregateTests, AvgTest)
{
    Config config;

    config.debug();

    const auto file = VOILA_TEST_SOURCES_PATH "/simple_avg.voila";
    constexpr size_t TENSOR_SIZE = 100;
    constexpr double TENSOR_VALS = 123;
    constexpr double TENSOR_AVG = TENSOR_VALS;
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<double[]>(new double[TENSOR_SIZE]);
    std::fill_n(arg.get(), TENSOR_SIZE, TENSOR_VALS);
    prog << ::voila::make_param(arg.get(), TENSOR_SIZE);

    // run in jit
    auto res = std::get<double>(prog()[0]);

    ASSERT_EQ(res, TENSOR_AVG);
}

TEST(AggregateTests, CountTest)
{
    Config config;

    config.debug();

    const auto file = VOILA_TEST_SOURCES_PATH "/simple_count.voila";
    constexpr size_t TENSOR_SIZE = 100;
    constexpr uint64_t TENSOR_VALS = 123;
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    std::fill_n(arg.get(), TENSOR_SIZE, TENSOR_VALS);
    prog << ::voila::make_param(arg.get(), TENSOR_SIZE);

    // run in jit
    auto res = std::get<uint64_t>(prog()[0]);

    ASSERT_EQ(res, TENSOR_SIZE);
}

TEST(AggregateTests, GroupSumTest)
{
    Config config;

    config.debug();

    const auto file = VOILA_TEST_SOURCES_PATH "/grouped_sum.voila";
    constexpr size_t TENSOR_SIZE = 100;
    constexpr size_t NEXT_POW = 128;
    constexpr uint64_t TENSOR_VALS = 123;
    constexpr uint64_t INDICES = 1;
    constexpr auto ref = std::to_array(
        {0, 12300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    auto arg2 = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    std::fill_n(arg.get(), TENSOR_SIZE, TENSOR_VALS);
    std::fill_n(arg2.get(), TENSOR_SIZE, INDICES);
    prog << ::voila::make_param(arg.get(), TENSOR_SIZE);
    prog << ::voila::make_param(arg2.get(), TENSOR_SIZE);

    // run in jit
    auto res = std::get<strided_memref_ptr<uint64_t, 1>>(prog()[0]);

    ASSERT_EQ(res->sizes[0], NEXT_POW);

    for (size_t i = 0; i < TENSOR_SIZE; ++i)
    {
        ASSERT_EQ(res->operator[](i), ref[i]);
    }
}

TEST(AggregateTests, GroupMinTest)
{
    Config config;

    config.debug();

    const auto file = VOILA_TEST_SOURCES_PATH "/grouped_min.voila";
    constexpr auto TENSOR_VALS = std::to_array<uint64_t>(
        {441, 965, 381, 125, 626, 162, 930, 213, 969, 866, 235, 571, 822, 469, 350, 73,  150, 494, 629, 236,
         15,  91,  843, 391, 771, 972, 759, 551, 388, 620, 651, 854, 810, 878, 737, 719, 331, 686, 532, 707,
         309, 540, 378, 872, 400, 8,   287, 38,  65,  452, 631, 337, 393, 502, 261, 917, 57,  410, 667, 561,
         431, 960, 64,  358, 488, 366, 820, 849, 529, 621, 890, 268, 230, 528, 87,  20,  117, 258, 794, 644,
         893, 565, 256, 906, 658, 557, 228, 176, 284, 159, 796, 18,  964, 635, 26,  105, 633, 832, 419, 369});
    constexpr uint64_t INDICES = 5;
    constexpr auto TENSOR_MIN = *std::min_element(TENSOR_VALS.begin(), TENSOR_VALS.end());
    constexpr auto MAX_VAL = std::numeric_limits<int64_t>::max();
    constexpr auto ref = std::to_array<uint64_t>(
        {MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, TENSOR_MIN, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL,
         MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL,    MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL,
         MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL,    MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL,
         MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL,    MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL,
         MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL,    MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL,
         MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL,    MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL,
         MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL,    MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL,
         MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL,    MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL,
         MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL,    MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL,
         MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL,    MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL,
         MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL, MAX_VAL,    MAX_VAL, MAX_VAL});
    constexpr size_t TENSOR_SIZE = 100;
    constexpr size_t NEXT_POW = 128;
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_VALS.size()]);
    auto arg2 = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    std::copy(TENSOR_VALS.begin(), TENSOR_VALS.end(), arg.get());
    std::fill_n(arg2.get(), TENSOR_SIZE, INDICES);
    prog << ::voila::make_param(arg.get(), TENSOR_VALS.size());
    prog << ::voila::make_param(arg2.get(), TENSOR_VALS.size());

    // run in jit
    auto res = std::get<strided_memref_ptr<uint64_t, 1>>(prog()[0]);

    ASSERT_EQ(res->sizes[0], NEXT_POW);

    for (size_t i = 0; i < NEXT_POW; ++i)
    {
        ASSERT_EQ(res->operator[](i), ref[i]);
    }
}

TEST(AggregateTests, GroupMaxTest)
{
    Config config;

    config.debug();

    const auto file = VOILA_TEST_SOURCES_PATH "/grouped_max.voila";
    constexpr auto TENSOR_VALS = std::to_array<int64_t>(
        {441, 965, 381, 125, 626, 162, 930, 213, 969, 866, 235, 571, 822, 469, 350, 73,  150, 494, 629, 236,
         15,  91,  843, 391, 771, 972, 759, 551, 388, 620, 651, 854, 810, 878, 737, 719, 331, 686, 532, 707,
         309, 540, 378, 872, 400, 8,   287, 38,  65,  452, 631, 337, 393, 502, 261, 917, 57,  410, 667, 561,
         431, 960, 64,  358, 488, 366, 820, 849, 529, 621, 890, 268, 230, 528, 87,  20,  117, 258, 794, 644,
         893, 565, 256, 906, 658, 557, 228, 176, 284, 159, 796, 18,  964, 635, 26,  105, 633, 832, 419, 369});
    constexpr uint64_t INDICES = 5;
    constexpr auto TENSOR_MAX = *std::max_element(TENSOR_VALS.begin(), TENSOR_VALS.end());
    constexpr auto MIN_VAL = std::numeric_limits<int64_t>::min();
    constexpr auto ref = std::to_array<int64_t>(
        {MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, TENSOR_MAX, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL,
         MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL,    MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL,
         MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL,    MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL,
         MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL,    MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL,
         MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL,    MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL,
         MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL,    MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL,
         MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL,    MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL,
         MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL,    MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL,
         MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL,    MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL,
         MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL,    MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL,
         MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL, MIN_VAL,    MIN_VAL, MIN_VAL});
    constexpr size_t TENSOR_SIZE = 100;
    constexpr size_t NEXT_POW = 128;
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_VALS.size()]);
    auto arg2 = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    std::copy(TENSOR_VALS.begin(), TENSOR_VALS.end(), arg.get());
    std::fill_n(arg2.get(), TENSOR_SIZE, INDICES);
    prog << ::voila::make_param(arg.get(), TENSOR_VALS.size());
    prog << ::voila::make_param(arg2.get(), TENSOR_VALS.size());

    // run in jit
    auto res = std::get<strided_memref_ptr<uint64_t, 1>>(prog()[0]);

    ASSERT_EQ(res->sizes[0], NEXT_POW);

    for (size_t i = 0; i < NEXT_POW; ++i)
    {
        ASSERT_EQ(res->operator[](i), ref[i]);
    }
}

TEST(AggregateTests, GroupAvgTest)
{
    Config config;

    config.debug();

    const auto file = VOILA_TEST_SOURCES_PATH "/grouped_avg.voila";
    constexpr size_t TENSOR_SIZE = 100;
    constexpr uint64_t TENSOR_VALS = 123;
    constexpr uint64_t TENSOR_AVG = TENSOR_VALS;
    constexpr uint64_t INDICES = 7;
    constexpr size_t NEXT_POW = 128;
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<double[]>(new double[TENSOR_SIZE]);
    auto arg2 = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    std::fill_n(arg.get(), TENSOR_SIZE, TENSOR_VALS);
    std::fill_n(arg2.get(), TENSOR_SIZE, INDICES);
    prog << ::voila::make_param(arg.get(), TENSOR_SIZE);
    prog << ::voila::make_param(arg2.get(), TENSOR_SIZE);

    // run in jit
    auto res = std::get<strided_memref_ptr<double, 1>>(prog()[0]);

    ASSERT_EQ(res->sizes[0], NEXT_POW);

    for (size_t i = 0; i < INDICES; ++i)
    {
        ASSERT_TRUE(std::isnan(res->operator[](i)));
    }
    ASSERT_EQ(res->operator[](INDICES), TENSOR_AVG);
    for (size_t i = INDICES + 1; i < NEXT_POW; ++i)
    {
        ASSERT_TRUE(std::isnan(res->operator[](i)));
    }
}

TEST(AggregateTests, GroupCountTest)
{
    Config config;

    config.debug();

    const auto file = VOILA_TEST_SOURCES_PATH "/grouped_count.voila";
    constexpr size_t TENSOR_SIZE = 100;
    constexpr uint64_t TENSOR_VALS = 123;
    constexpr uint64_t INDICES = 3;
    constexpr size_t NEXT_POW = 128;
    constexpr auto ref = std::to_array({0, 0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    auto arg2 = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    std::fill_n(arg.get(), TENSOR_SIZE, TENSOR_VALS);
    std::fill_n(arg2.get(), TENSOR_SIZE, INDICES);
    prog << ::voila::make_param(arg.get(), TENSOR_SIZE);
    prog << ::voila::make_param(arg2.get(), TENSOR_SIZE);

    // run in jit
    auto res = std::get<strided_memref_ptr<uint64_t, 1>>(prog()[0]);

    ASSERT_EQ(res->sizes[0], NEXT_POW);

    for (size_t i = 0; i < NEXT_POW; ++i)
    {
        ASSERT_EQ(res->operator[](i), ref[i]);
    }
}
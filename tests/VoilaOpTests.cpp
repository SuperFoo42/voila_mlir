#include "Config.hpp"
#include "Program.hpp"
#include "test_defs.hpp.inc"

#include <gtest/gtest.h>
using namespace voila;

TEST(AddTests, TensorTensorTest)
{
    Config config;

    config.debug = true;
    config.optimize = false;

    const auto file = VOILA_TEST_SOURCES_PATH "/simple_add.voila";
    constexpr size_t TENSOR_SIZE = 100;
    constexpr uint64_t TENSOR_VALS = 123;
    constexpr uint64_t TENSOR_SUM = TENSOR_VALS + TENSOR_VALS;
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    std::fill_n(arg.get(), TENSOR_SIZE, TENSOR_VALS);
    auto arg2 = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    std::fill_n(arg2.get(), TENSOR_SIZE, TENSOR_VALS);
    prog << ::voila::make_param(arg.get(), TENSOR_SIZE, voila::DataType::INT64);
    prog << ::voila::make_param(arg2.get(), TENSOR_SIZE, voila::DataType::INT64);

    // run in jit
    auto res = prog();

    ASSERT_EQ((*std::get<std::unique_ptr<StridedMemRefType<uint64_t, 1> *>>(res))->sizes[0], TENSOR_SIZE);
    for (auto elem : *(*std::get<std::unique_ptr<StridedMemRefType<uint64_t, 1> *>>(res)))
        ASSERT_EQ(elem, TENSOR_SUM);
}

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
    auto res = prog();

    ASSERT_EQ((*std::get<std::unique_ptr<StridedMemRefType<uint64_t, 1> *>>(res))->sizes[0], TENSOR_SIZE);
    for (auto elem : *(*std::get<std::unique_ptr<StridedMemRefType<uint64_t, 1> *>>(res)))
        ASSERT_EQ(elem, TENSOR_SUM);
}

TEST(AddTests, ScalarTensorTest)
{
    FAIL();
}

TEST(AddTests, ScalarScalarTest)
{
    FAIL();
}

TEST(SubTests, TensorTensorTest)
{
    Config config;

    config.debug = true;
    config.optimize = true;

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
    prog << ::voila::make_param(arg.get(), TENSOR_SIZE, voila::DataType::INT64);
    prog << ::voila::make_param(arg2.get(), TENSOR_SIZE, voila::DataType::INT64);

    // run in jit
    auto res = prog();

    ASSERT_EQ((*std::get<std::unique_ptr<StridedMemRefType<uint64_t, 1> *>>(res))->sizes[0], TENSOR_SIZE);
    for (auto elem : *(*std::get<std::unique_ptr<StridedMemRefType<uint64_t, 1> *>>(res)))
        ASSERT_EQ(elem, TENSOR_SUB);
}

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
}

TEST(HashTableTests, Hash)
{
    Config config;

    config.debug = true;
    config.optimize = true;
    const auto file = VOILA_TEST_SOURCES_PATH "/simple_hash.voila";
    constexpr size_t TENSOR_SIZE = 100;
    constexpr uint64_t TENSOR_VALS = 123;
    constexpr uint64_t HASH = 7668608003591710536;
    /*constexpr auto ref = std::to_array({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 123, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0});*/
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    std::fill_n(arg.get(), TENSOR_SIZE, TENSOR_VALS);
    prog << ::voila::make_param(arg.get(), TENSOR_SIZE, voila::DataType::INT64);

    // run in jit
    auto res = prog();

    ASSERT_EQ((*std::get<std::unique_ptr<StridedMemRefType<uint64_t, 1> *>>(res))->sizes[0], TENSOR_SIZE);

    for (size_t i = 0; i < TENSOR_SIZE; ++i)
    {
        ASSERT_EQ((*std::get<std::unique_ptr<StridedMemRefType<uint64_t, 1> *>>(res))->operator[](i), HASH);
    }
}

TEST(HashTableTests, Insert)
{
    Config config;

    config.debug = true;
    config.optimize = true;
    const auto file = VOILA_TEST_SOURCES_PATH "/simple_insert.voila";
    constexpr size_t TENSOR_SIZE = 100;
    constexpr size_t NEXTPOW = 128;
    constexpr uint64_t TENSOR_VALS = 123;
    constexpr auto ref = std::to_array({0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 123, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0, 0, 0,
                                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   0, 0, 0, 0, 0, 0});
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    std::fill_n(arg.get(), TENSOR_SIZE, TENSOR_VALS);
    prog << ::voila::make_param(arg.get(), TENSOR_SIZE, voila::DataType::INT64);

    // run in jit
    auto res = prog();

    ASSERT_EQ((*std::get<std::unique_ptr<StridedMemRefType<uint64_t, 1> *>>(res))->sizes[0], NEXTPOW);

    for (size_t i = 0; i < TENSOR_SIZE; ++i)
    {
        ASSERT_EQ((*std::get<std::unique_ptr<StridedMemRefType<uint64_t, 1> *>>(res))->operator[](i), ref[i]);
    }
}

TEST(HashTableTests, Lookup)
{
    Config config;

    config.debug = true;
    config.optimize = true;
    const auto file = VOILA_TEST_SOURCES_PATH "/simple_lookup.voila";
    constexpr size_t TENSOR_SIZE = 100;
    constexpr uint64_t TENSOR_VALS = 123;
    constexpr size_t VALUE_POS = 72;
    Program prog(file, config);
    // alloc dummy data to pass to program args
    auto arg = std::unique_ptr<uint64_t[]>(new uint64_t[TENSOR_SIZE]);
    std::fill_n(arg.get(), TENSOR_SIZE, TENSOR_VALS);
    prog << ::voila::make_param(arg.get(), TENSOR_SIZE, voila::DataType::INT64);

    // run in jit
    auto res = prog();

    ASSERT_EQ((*std::get<std::unique_ptr<StridedMemRefType<uint64_t, 1> *>>(res))->sizes[0], TENSOR_SIZE);

    for (size_t i = 0; i < TENSOR_SIZE; ++i)
    {
        ASSERT_EQ((*std::get<std::unique_ptr<StridedMemRefType<uint64_t, 1> *>>(res))->operator[](i), VALUE_POS);
    }
}
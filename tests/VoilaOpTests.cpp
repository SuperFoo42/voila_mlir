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

    const auto file = VOILA_TEST_SOURCES_PATH"/simple_add.voila";
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

TEST(AddTests, TensorScalarTest) {
    Config config;

    config.debug = true;
    config.optimize = false;

    const auto file = VOILA_TEST_SOURCES_PATH"/simple_add.voila";
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

TEST(AddTests, ScalarTensorTest) {}

TEST(AddTests, ScalarScalarTest) {}

TEST(SubTests, TensorTensorTest) {
    Config config;

    config.debug = true;
    config.optimize = false;

    const auto file = VOILA_TEST_SOURCES_PATH"/simple_sub.voila";
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

TEST(SubTests, TensorScalarTest) {}

TEST(SubTests, ScalarTensorTest) {}

TEST(SubTests, ScalarScalarTest) {}
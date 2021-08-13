#pragma once
#include "fmt/format.h"

#include <magic_enum.hpp>
#include <ostream>
#include <utility>
namespace voila
{
    enum class DataType
    {
        BOOL,
        NUMERIC,
        INT32,
        INT64,
        DBL,
        STRING,
        VOID,
        UNKNOWN
    };

    class Arity
    {
        size_t arity;
        bool undef;

      public:
        Arity();
        friend std::ostream &operator<<(std::ostream &os, const Arity &ar);
        bool operator==(const Arity &rhs) const;
        bool operator!=(const Arity &rhs) const;

        [[nodiscard]] bool is_undef() const;

        [[nodiscard]] size_t get_size() const;

        explicit Arity(const size_t i)
        {
            arity = i;
            undef = false;
        }
    };

    class Type
    {
      public:
        static bool convertibleDataTypes(DataType t1, DataType t2);
        [[maybe_unused]] size_t typeID;
        virtual ~Type() = default;
        explicit Type(size_t tID) : typeID(tID) {}
        [[nodiscard]] virtual bool convertible(const Type &) const = 0;
        [[nodiscard]] virtual bool convertible(const DataType &) const = 0;
        [[nodiscard]] virtual std::vector<DataType> getTypes() const = 0;
        [[nodiscard]] virtual std::vector<Arity> getArities() const = 0;
        [[nodiscard]] bool compatible(const Type &other) const
        {
            return convertible(other) || other.convertible(*this);
        }

        [[nodiscard]] virtual bool compatible(const DataType &other) const = 0;
    };

    class ScalarType;
    class FunctionType;

    class ScalarType : public Type
    {
      public:
        [[maybe_unused]] DataType t;
        [[maybe_unused]] Arity ar;

        explicit ScalarType(size_t tID, DataType t = DataType::UNKNOWN, Arity ar = Arity());
        friend std::ostream &operator<<(std::ostream &os, const ScalarType &type);

        [[nodiscard]] bool convertible(const Type &other) const override;
        [[nodiscard]] bool convertible(const DataType &type) const override;
        [[nodiscard]] bool compatible(const DataType &other) const override;
        [[nodiscard]] std::vector<DataType> getTypes() const override;
        [[nodiscard]] std::vector<Arity> getArities() const override;
    };

    class FunctionType : public Type
    {
      public:
        explicit FunctionType(size_t tID,
                              std::vector<size_t> paramTypeIds = {},
                              std::vector<std::pair<DataType, Arity>> returnTypes = {
                                  std::make_pair(DataType::UNKNOWN, Arity())});

        friend std::ostream &operator<<(std::ostream &os, const FunctionType &type);
        [[nodiscard]] bool convertible(const Type &other) const override;
        [[nodiscard]] bool convertible(const DataType &type) const override;
        [[nodiscard]] bool compatible(const DataType &other) const override;
        [[nodiscard]] std::vector<DataType> getTypes() const override;
        [[nodiscard]] std::vector<Arity> getArities() const override;
        std::vector<size_t> paramTypeIds;
        std::vector<std::pair<DataType, Arity>> returnTypes;
    };
} // namespace voila
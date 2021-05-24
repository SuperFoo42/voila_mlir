#pragma once
#include <magic_enum.hpp>
#include <ostream>
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
        [[maybe_unused]] size_t arity;
        [[maybe_unused]] bool undef;

      public:
        Arity() : arity{std::numeric_limits<decltype(arity)>::max()}, undef{true} {}
        friend std::ostream &operator<<(std::ostream &os, const Arity &ar)
        {
            if (ar.undef)
                os << "undef";
            else
                os << std::to_string(ar.arity);

            return os;
        }
        bool operator==(const Arity &rhs) const
        {
            return arity == rhs.arity || undef == rhs.undef;
        }
        bool operator!=(const Arity &rhs) const
        {
            return !(rhs == *this);
        }

        bool is_undef() const
        {
            return undef;
        }

        size_t get_size() const
        {
            return arity;
        }
    };

    class Type
    {
      public:
        [[maybe_unused]] size_t typeID;
        [[maybe_unused]] DataType t;
        [[maybe_unused]] Arity ar;
        Type() = default;
        virtual ~Type() = default;
        explicit Type(size_t tID, DataType t = DataType::UNKNOWN, Arity ar = Arity()) : typeID(tID), t(t), ar{ar} {}
        friend std::ostream &operator<<(std::ostream &os, const Type &type)
        {
            os << fmt::format("T{}:{}[", type.typeID, std::string(magic_enum::enum_name(type.t)));
            os << type.ar << "]";
            return os;
        }
    };

    class FunctionType : public Type
    {
        std::vector<size_t> paramTypeIds;

      public:
        explicit FunctionType(size_t tID,
                              std::vector<size_t> paramTypeIds = {},
                              DataType t = DataType::UNKNOWN,
                              Arity ar = Arity()) :
            Type(tID, t, ar), paramTypeIds{std::move(paramTypeIds)}
        {
        }

        friend std::ostream &operator<<(std::ostream &os, const FunctionType &type)
        {
            os << fmt::format("T{}:{}[", type.typeID, std::string(magic_enum::enum_name(type.t)));
            os << type.ar << "](";
            os << fmt::format("T{}", fmt::join(type.paramTypeIds, ", T"));
            os << ")";
            return os;
        }
    };
} // namespace voila
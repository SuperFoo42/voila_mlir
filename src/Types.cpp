#include "Types.hpp"

#include <vector>
namespace voila
{
    std::ostream &operator<<(std::ostream &os, const Arity &ar)
    {
        if (ar.undef)
            os << "undef";
        else
            os << std::to_string(ar.arity);

        return os;
    }
    bool Arity::operator==(const Arity &rhs) const
    {
        return arity == rhs.arity || undef == rhs.undef;
    }
    bool Arity::operator!=(const Arity &rhs) const
    {
        return !(rhs == *this);
    }
    bool Arity::is_undef() const
    {
        return undef;
    }
    size_t Arity::get_size() const
    {
        return arity;
    }
    Arity::Arity() : arity{std::numeric_limits<decltype(arity)>::max()}, undef{true} {}
    std::ostream &operator<<(std::ostream &os, const ScalarType &type)
    {
        os << fmt::format("T{}:{}[", type.typeID, std::string(magic_enum::enum_name(type.t)));
        os << type.ar << "]";
        return os;
    }
    ScalarType::ScalarType(size_t tID, DataType t, Arity ar) : Type(tID), t(t), ar{ar} {}

    bool ScalarType::convertible(const Type &other) const
    {
        if (dynamic_cast<const ScalarType *>(&other))
        {
            auto scalarType = dynamic_cast<const ScalarType &>(other);
            return convertibleDataTypes(this->t, scalarType.t);
        }
        else
        {
            auto functionType = dynamic_cast<const FunctionType &>(other);
            if (functionType.returnTypes.size() != 1)
                return false;
            else
            {
                return convertibleDataTypes(this->t, functionType.returnTypes.front().first);
            }
        }
    }
    bool ScalarType::convertible(const DataType &type) const
    {
        return convertibleDataTypes(t, type);
    }
    bool ScalarType::compatible(const DataType &other) const
    {
        return convertible(other) || convertibleDataTypes(other, t);
    }
    std::vector<DataType> ScalarType::getTypes() const
    {
        return {t};
    }
    std::vector<Arity> ScalarType::getArities() const
    {
        return {ar};
    }

    std::ostream &operator<<(std::ostream &os, const FunctionType &type)
    {
        os << fmt::format("T{}:(", type.typeID);
        for (const auto &t : type.returnTypes)
        {
            os << std::string(magic_enum::enum_name(t.first)) << "[" << t.second << "],"; // TODO: one comma to much
        }
        os << ")(";
        os << fmt::format("T{}", fmt::join(type.paramTypeIds, ", T"));
        os << ")";
        return os;
    }
    FunctionType::FunctionType(size_t tID,
                               std::vector<size_t> paramTypeIds,
                               std::vector<std::pair<DataType, Arity>> returnTypes) :
        Type(tID), paramTypeIds{std::move(paramTypeIds)}, returnTypes{std::move(returnTypes)}
    {
    }

    bool FunctionType::convertible(const Type &other) const
    {
        if (dynamic_cast<const ScalarType *>(&other))
        {
            auto scalarType = dynamic_cast<const ScalarType &>(other);

            if (returnTypes.size() != 1)
                return false;
            else
            {
                return convertibleDataTypes(returnTypes.front().first, scalarType.t);
            }
        }
        else
        {
            auto functionType = dynamic_cast<const FunctionType &>(other);
            if (this->returnTypes.size() != returnTypes.size())
                return false;
            bool all_eq = true;
            for (size_t i = 0; i < returnTypes.size(); ++i)
            {
                all_eq &= convertibleDataTypes(returnTypes.at(i).first, functionType.returnTypes.at(i).first);
            }
            return all_eq;
        }
    }
    bool FunctionType::convertible(const DataType &type) const
    {
        if (returnTypes.size() != 1)
            return false;
        else
        {
            return convertibleDataTypes(returnTypes.front().first, type);
        }
    }
    bool FunctionType::compatible(const DataType &other) const
    {
        if (returnTypes.size() == 1)
            return this->convertible(other) || convertibleDataTypes(other, returnTypes.front().first);

        return false;
    }
    std::vector<DataType> FunctionType::getTypes() const
    {
        std::vector<DataType> types;
        for (const auto &t : returnTypes)
            types.push_back(t.first);

        return types;
    }
    std::vector<Arity> FunctionType::getArities() const
    {
        std::vector<Arity> types;
        for (const auto &t : returnTypes)
            types.push_back(t.second);

        return types;
    }
    bool Type::convertibleDataTypes(DataType t1, DataType t2)
    {
        if (t1 == t2)
            return true;
        switch (t1)
        {
            case DataType::UNKNOWN:
                return true;
            case DataType::NUMERIC:
                return t2 == DataType::INT32 || t2 == DataType::INT64 || t2 == DataType::DBL || t2 == DataType::BOOL;
            case DataType::BOOL:
                return t2 == DataType::INT64 || t2 == DataType::INT32;
            case DataType::INT32:
                return t2 == DataType::INT64 || t2 == DataType::DBL;
            case DataType::INT64:
                return t2 == DataType::DBL;
            default:
                return false;
        }
    }
} // namespace voila
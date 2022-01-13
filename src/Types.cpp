#include "Types.hpp"

#include "TypeInferer.hpp"

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
    ScalarType::ScalarType(type_id_t tID, TypeInferer &inferer, DataType t, Arity ar) : Type(tID, inferer), t(t), ar{ar}
    {
    }

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
            if (functionType.returnTypeIDs.size() != 1)
                return false;
            else
            {
                return convertibleDataTypes(this->t,
                                            inferer.types.at(functionType.returnTypeIDs.front())->getTypes().front());
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
    std::vector<Arity> ScalarType::setAritiy(size_t idx, Arity arity)
    {
        if (idx != 0)
        {
            throw std::runtime_error("Out of Bounds");
        }
        ar = arity;
    }

    std::ostream &operator<<(std::ostream &os, const FunctionType &type)
    {
        os << fmt::format("T{}:(", type.typeID);
        for (const auto &t : type.returnTypeIDs)
        {
            os << std::string(magic_enum::enum_name(dynamic_cast<ScalarType *>(type.inferer.types.at(t).get())->t))
               << "[" << dynamic_cast<ScalarType *>(type.inferer.types.at(t).get())->ar
               << "],"; // TODO: one comma to much
        }
        os << ")(";
        os << fmt::format("T{}", fmt::join(type.paramTypeIds, ", T"));
        os << ")";
        return os;
    }
    FunctionType::FunctionType(type_id_t tID,
                               TypeInferer &inferer,
                               std::vector<type_id_t> paramTypeIds,
                               std::vector<type_id_t> returnTypes) :
        Type(tID, inferer), paramTypeIds{std::move(paramTypeIds)}, returnTypeIDs{std::move(returnTypes)}
    {
    }

    bool FunctionType::convertible(const Type &other) const
    {
        if (dynamic_cast<const ScalarType *>(&other))
        {
            auto scalarType = dynamic_cast<const ScalarType &>(other);

            if (returnTypeIDs.size() != 1)
                return false;
            else
            {
                return convertibleDataTypes(
                    dynamic_cast<ScalarType *>(inferer.types.at(returnTypeIDs.front()).get())->t, scalarType.t);
            }
        }
        else
        {
            auto functionType = dynamic_cast<const FunctionType &>(other);
            if (this->returnTypeIDs.size() != returnTypeIDs.size())
                return false;
            bool all_eq = true;
            for (size_t i = 0; i < returnTypeIDs.size(); ++i)
            {
                all_eq &= convertibleDataTypes(
                    dynamic_cast<ScalarType *>(inferer.types.at(returnTypeIDs.at(i)).get())->t,
                    dynamic_cast<ScalarType *>(inferer.types.at(functionType.returnTypeIDs.at(i)).get())->t);
            }
            return all_eq;
        }
    }
    bool FunctionType::convertible(const DataType &type) const
    {
        if (returnTypeIDs.size() != 1)
            return false;
        else
        {
            return convertibleDataTypes(dynamic_cast<ScalarType *>(inferer.types.at(returnTypeIDs.front()).get())->t,
                                        type);
        }
    }
    bool FunctionType::compatible(const DataType &other) const
    {
        if (returnTypeIDs.size() == 1)
            return this->convertible(other) ||
                   convertibleDataTypes(other,
                                        dynamic_cast<ScalarType *>(inferer.types.at(returnTypeIDs.front()).get())->t);

        return false;
    }
    std::vector<DataType> FunctionType::getTypes() const
    {
        std::vector<DataType> types;
        for (const auto &t : returnTypeIDs)
        {
            auto tmp = inferer.types.at(t)->getTypes();
            types.insert(types.end(), tmp.begin(), tmp.end());
        }

        return types;
    }
    std::vector<Arity> FunctionType::getArities() const
    {
        std::vector<Arity> types;
        for (const auto &t : returnTypeIDs)
        {
            auto tmp = inferer.types.at(t)->getArities();
            types.insert(types.end(), tmp.begin(), tmp.end());
        }

        return types;
    }
    std::vector<Arity> FunctionType::setAritiy(size_t idx, Arity ar)
    {
        return std::vector<Arity>();
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
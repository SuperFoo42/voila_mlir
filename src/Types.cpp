#include "Types.hpp"
#include "TypeInferer.hpp"                  // for TypeInferer
#include "magic_enum.hpp"                   // for enable_if_enum_t, enum_name
#include "range/v3/algorithm/all_of.hpp"    // for all_of, all_of_fn
#include "range/v3/functional/identity.hpp" // for identity
#include "llvm/ADT/iterator_range.h"        // for make_range
#include "llvm/Support/FormatVariadic.h"    // for formatv, formatv_object
#include <utility>                          // for move
#include <vector>                           // for vector

namespace voila
{
    using std::to_string;

    std::ostream &operator<<(std::ostream &os, const Arity &ar)
    {
        os << to_string(ar);

        return os;
    }

    std::string to_string(const Arity &ar)
    {
        if (ar.undef())
            return "undef";

        return std::to_string(ar.arity);
    }

    bool Arity::operator==(const Arity &rhs) const { return arity == rhs.arity; }

    bool Arity::operator!=(const Arity &rhs) const { return !(rhs == *this); }

    bool Arity::undef() const { return arity == UNDEF; }

    size_t Arity::get_size() const { return arity; }

    std::ostream &operator<<(std::ostream &os, const ScalarType &type)
    {
        os << llvm::formatv("T{0}:{1}[", type.typeID, to_string(type.t)).str();
        os << type.ar << "]";
        return os;
    }

    bool ScalarType::undef() const { return t == DataType::UNKNOWN && ar.undef(); }

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

    bool ScalarType::convertible(const DataType &type) const { return convertibleDataTypes(t, type); }

    bool ScalarType::compatible(const DataType &other) const
    {
        return convertible(other) || convertibleDataTypes(other, t);
    }

    std::vector<DataType> ScalarType::getTypes() const { return {t}; }

    std::vector<std::reference_wrapper<Arity>> ScalarType::getArities() { return {ar}; }

    std::vector<Arity> ScalarType::getArities() const { return {ar}; }

    std::string ScalarType::stringify() const { return to_string(t) + to_string(ar); }

    std::ostream &operator<<(std::ostream &os, const FunctionType &type)
    {
        os << llvm::formatv("T{0}:(", type.typeID).str();
        for (const auto &t : type.returnTypeIDs)
        {
            os << to_string(dynamic_cast<ScalarType *>(type.inferer.types.at(t).get())->t) << "["
               << dynamic_cast<ScalarType *>(type.inferer.types.at(t).get())->ar << "],"; // TODO: one comma to much
        }
        os << ")(";
        os << llvm::formatv("{0:@[T]}", llvm::make_range(type.paramTypeIds.begin(), type.paramTypeIds.end())).str();
        os << ")";
        return os;
    }

    FunctionType::FunctionType(type_id_t tID,
                               TypeInferer &inferer,
                               std::vector<type_id_t> paramTypeIds,
                               std::vector<type_id_t> returnTypes)
        : Type(tID, inferer), paramTypeIds{std::move(paramTypeIds)}, returnTypeIDs{std::move(returnTypes)}
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

    std::vector<std::reference_wrapper<Arity>> FunctionType::getArities()
    {
        std::vector<std::reference_wrapper<Arity>> types;
        for (const auto &t : returnTypeIDs)
        {
            auto tmp = inferer.types.at(t)->getArities();
            types.insert(types.end(), tmp.begin(), tmp.end());
        }

        return types;
    }

    std::string to_string(const Type &t) { return t.stringify(); }

    std::string to_string(const DataType &dt) { return std::string(magic_enum::enum_name(dt)); }

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

    std::string FunctionType::stringify() const
    {
        std::string type;
        for (const auto &t : paramTypeIds)
        {
            auto tpe = inferer.types.at(t);
            if (dynamic_cast<FunctionType *>(tpe.get()))
                type += std::static_pointer_cast<FunctionType>(tpe)->stringify_rets() + "_";
            else
                type += to_string(*tpe) + "_";
        }
        type += "ret_";
        /*        for (const auto &t: returnTypeIDs) {
                    type += stringify_rets();
                }*/

        return type;
    }

    std::string FunctionType::stringify_rets() const
    {
        std::string type;
        for (auto rid : returnTypeIDs)
        {
            auto t = inferer.types.at(rid);
            if (dynamic_cast<FunctionType *>(t.get()))
                type += std::static_pointer_cast<FunctionType>(t)->stringify_rets() + "_";
            else
                type += to_string(*t) + "_";
        }

        return type;
    }

    bool FunctionType::undef() const
    {
        return ranges::all_of(paramTypeIds, [&](auto t) { return inferer.types.at(t)->undef(); }) &&
               ranges::all_of(returnTypeIDs, [&](auto id) { return inferer.types.at(id)->undef(); });
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
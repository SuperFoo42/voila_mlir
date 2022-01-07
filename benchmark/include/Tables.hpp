#pragma once
#include "ColumnTypes.hpp"
#include "DateReformatter.hpp"
#include "DictionaryCompressor.hpp"

#include <bit>
#include <cassert>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/variant.hpp>
#include <cereal/types/vector.hpp>
#include <fstream>
#include <string>
#include <variant>
#include <vector>
class Table
{
  public:
    static Table readTable(const std::string &path)
    {
        std::ifstream file(path, std::ios::binary);
        cereal::PortableBinaryInputArchive inputArchive(file);
        Table tbl;
        inputArchive(tbl);
        return tbl;
    }

    static void writeTable(const std::string &path, const Table &tbl)
    {
        std::ofstream file(path, std::ios::binary);
        cereal::PortableBinaryOutputArchive outputArchive(file);
        outputArchive(tbl);
    }

    using column_type = std::variant<std::vector<std::string>, std::vector<double>, std::vector<int32_t>>;
    using row_type = std::variant<std::string, double, int32_t>;
    virtual void addColumn(column_type &&column, ColumnTypes type)
    {
        columns.emplace_back(column);
        types.push_back(type);
    }

    virtual std::vector<row_type> getRow(size_t i)
    {
        assert(i < rowCount());
        std::vector<row_type> row;
        row.reserve(rowCount());
        std::transform(
            columns.begin(), columns.end(), std::back_inserter(row), [&i](const auto &c) -> auto {
                return std::visit([&](const auto &val) -> row_type { return val.at(i); }, c);
            });
        return row;
    }

    size_t columnCount()
    {
        return columns.size();
    }

    size_t rowCount()
    {
        return std::visit(
            [](const auto &col) -> auto { return col.size(); }, columns.at(0));
    }

    template<class R>
    std::vector<R> &getColumn(const size_t idx)
    {
        return std::get<std::vector<R>>(columns.at(idx));
    }

    virtual void addRow(std::vector<row_type> &row)
    {
        assert(row.size() == columns.size());
        for (size_t i = 0; i < columns.size(); ++i)
        {
            switch (types.at(i))
            {
                case ColumnTypes::INT:
                    std::visit(
                        [&](auto v) -> void
                        {
                            using T = std::decay_t<decltype(v)>;
                            if constexpr (std::is_same_v<T, int32_t>)
                                this->getColumn<int32_t>(i).push_back(v);
                            else
                                assert(false && "non-exhaustive visitor!");
                        },
                        row.at(i));
                    break;
                case ColumnTypes::STRING:
                    std::visit(
                        [&](auto v) -> void
                        {
                            using T = std::decay_t<decltype(v)>;
                            if constexpr (std::is_same_v<T, std::string>)
                                this->getColumn<std::string>(i).push_back(v);
                            else
                                assert(false && "non-exhaustive visitor!");
                        },
                        row.at(i));
                    break;
                case ColumnTypes::DATE:
                    std::visit(
                        [&](auto v) -> void
                        {
                            using T = std::decay_t<decltype(v)>;
                            if constexpr (std::is_same_v<T, int32_t>)
                                this->getColumn<int32_t>(i).push_back(v);
                            else
                                assert(false && "non-exhaustive visitor!");
                        },
                        row.at(i));
                    break;
                case ColumnTypes::DECIMAL:
                    std::visit(
                        [&](auto v) -> void
                        {
                            using T = std::decay_t<decltype(v)>;
                            if constexpr (std::is_same_v<T, double>)
                                this->getColumn<double>(i).push_back(v);
                            else
                                assert(false && "non-exhaustive visitor!");
                        },
                        row.at(i));
                    break;
            }
        }
    }

    virtual Table join(Table &other, size_t joinCol1, size_t joinCol2)
    {
        Table t;
        t.columns.resize(columnCount() + other.columnCount());
        t.types.insert(t.types.end(), types.begin(), types.end());
        t.types.insert(t.types.end(), other.types.begin(), other.types.end());

        for (size_t i = 0; i < rowCount(); ++i)
        {
            for (size_t j = 0; j < other.rowCount(); ++j)
            {
                auto match = [&]() -> bool
                {
                    if (types.at(joinCol1) != other.types.at(joinCol2))
                        return false;

                    switch (types.at(joinCol1))
                    {
                        case ColumnTypes::INT:
                            return getColumn<int32_t>(joinCol1).at(i) == other.getColumn<int32_t>(joinCol2).at(j);
                        case ColumnTypes::DECIMAL:
                            return getColumn<double>(joinCol1).at(i) == other.getColumn<double>(joinCol2).at(j);
                        case ColumnTypes::STRING:
                            return getColumn<std::string>(joinCol1).at(i) ==
                                   other.getColumn<std::string>(joinCol1).at(j);
                        case ColumnTypes::DATE:
                            return getColumn<int32_t>(joinCol1).at(i) == other.getColumn<int32_t>(joinCol2).at(j);
                    }
                };
                if (match())
                {
                    auto tmp = getRow(i);
                    auto otmp = other.getRow(j);
                    tmp.insert(tmp.end(), otmp.begin(), otmp.end());
                    t.addRow(tmp);
                }
            }
        }
        return t;
    }

    static Table makeWideTable(std::vector<Table> tables, std::vector<std::pair<size_t, size_t>> joinCols)
    {
        size_t offset = 0;
        Table wideTable(tables.front());
        for (size_t i = 1; i < tables.size(); ++i)
        {
            auto cols = joinCols.at(i);
            wideTable = wideTable.join(tables.at(i), cols.first + offset, cols.second);
            offset += tables.at(i - 1).rowCount();
        }

        return wideTable;
    }

    virtual ~Table() = default;

    template<class Archive>
    void serialize(Archive &ar)
    {
        ar(columns, types);
    }

  protected:
    std::vector<column_type> columns;
    std::vector<ColumnTypes> types;
};

class CompressedTable : public Table
{
    std::vector<std::unordered_map<std::string, int32_t>> dictionaries;

  public:
    ~CompressedTable() override = default;
    void addColumn(column_type &&column, ColumnTypes type) final
    {
        if (type == ColumnTypes::DATE)
        {
            DateReformatter reformatter(std::move(std::get<std::vector<std::string>>(column)));
            Table::addColumn(reformatter.reformat(), type);
        }
        else if (type == ColumnTypes::STRING)
        {
            DictionaryCompressor compressor(std::move(std::get<std::vector<std::string>>(column)));
            auto compressedCol = compressor.compress();
            dictionaries.resize(columnCount());
            dictionaries.emplace_back(compressedCol.second);
            Table::addColumn(compressedCol.first, type);
        }
        else
        {
            Table::addColumn(std::move(column), type);
        }
    }

    template<class Archive>
    void serialize(Archive &ar)
    {
        ar(cereal::base_class<Table>(this), dictionaries);
    }

    auto getDictionary(size_t column)
    {
        return dictionaries.at(column);
    }

    static CompressedTable readTable(const std::string &path)
    {
        std::ifstream file(path, std::ios::binary);
        cereal::PortableBinaryInputArchive inputArchive(file);
        Table tbl;
        inputArchive(tbl);
        return dynamic_cast<CompressedTable &>(tbl);
    }
};
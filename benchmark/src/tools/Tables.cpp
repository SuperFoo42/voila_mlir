#include "Tables.hpp"

#include <bit>
#include <bxzstr.hpp>
#include <cassert>
#include <filesystem>
#include <fstream>

ColumnTypes Table::getColType(size_t col)
{
    return types.at(col);
}
void Table::addColumn(Table::column_type &&column, ColumnTypes type)
{
    columns.emplace_back(column);
    types.push_back(type);
}
std::vector<Table::row_type> Table::getRow(size_t i)
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
size_t Table::columnCount()
{
    return columns.size();
}
size_t Table::rowCount()
{
    return std::visit(
        [](const auto &col) -> auto { return col.size(); }, columns.at(0));
}
void Table::addRow(std::vector<row_type> &row)
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
                        if constexpr (std::is_same_v<T, std::string>)
                            this->getColumn<std::string>(i).push_back(v);
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
Table Table::join(Table &other, size_t joinCol1, size_t joinCol2)
{
    Table t;
    t.columns.reserve(columnCount() + other.columnCount());
    t.types = types;
    t.types.insert(t.types.end(), other.types.begin(), other.types.end());

    for (const auto &tpe : t.types)
    {
        switch (tpe)
        {
            case ColumnTypes::INT:
                t.columns.emplace_back(std::vector<int32_t>());
                break;
            case ColumnTypes::STRING:
                t.columns.emplace_back(std::vector<std::string>());
                break;
            case ColumnTypes::DATE:
                t.columns.emplace_back(std::vector<std::string>());
                break;
            case ColumnTypes::DECIMAL:
                t.columns.emplace_back(std::vector<double>());
                break;
        }
    }

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
                        return getColumn<std::string>(joinCol1).at(i) == other.getColumn<std::string>(joinCol1).at(j);
                    case ColumnTypes::DATE:
                        return getColumn<int32_t>(joinCol1).at(i) == other.getColumn<int32_t>(joinCol2).at(j);
                }
                return false;
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
Table Table::makeWideTable(std::vector<Table> tables, std::vector<std::pair<size_t, size_t>> joinCols)
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

CompressedTable::CompressedTable(const std::string &path)
{
    if (!std::filesystem::exists(path))
        throw std::runtime_error("Could not open file");
    bxz::ifstream decompressor(path);
    cereal::PortableBinaryInputArchive inputArchive(decompressor);
    inputArchive(*this);
}

CompressedTable::CompressedTable(Table &&tbl)
{
    for (size_t i = 0; i < tbl.columnCount(); ++i)
    {
        switch (tbl.getColType(i))
        {
            case ColumnTypes::INT:
                addColumn(tbl.getColumn<int32_t>(i), tbl.getColType(i));
                break;
            case ColumnTypes::STRING:
                addColumn(tbl.getColumn<std::string>(i), tbl.getColType(i));
                break;
            case ColumnTypes::DATE:
                addColumn(tbl.getColumn<std::string>(i), tbl.getColType(i));
                break;
            case ColumnTypes::DECIMAL:
                addColumn(tbl.getColumn<double>(i), tbl.getColType(i));
                break;
        }
    }
}

void CompressedTable::addColumn(Table::column_type &&column, ColumnTypes type)
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
std::unordered_map<std::string, int32_t> CompressedTable::getDictionary(size_t column)
{
    return dictionaries.at(column);
}

void CompressedTable::writeTable(const std::string &path)
{
    bxz::ofstream compressor(path, bxz::lzma, 9);
    cereal::PortableBinaryOutputArchive outputArchive(compressor);
    outputArchive(*this);
}
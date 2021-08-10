#pragma once
#include "ColumnTypes.hpp"
#include "DateReformatter.hpp"
#include "DictionaryCompressor.hpp"

#include <bit>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/base_class.hpp>
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
    virtual void addColumn(column_type &&column, ColumnTypes type)
    {
        columns.emplace_back(column);
        types.push_back(type);
    }

    size_t columnCount()
    {
        return columns.size();
    }

    size_t rowCount()
    {
        // TODO
        return 0;
    }

    virtual ~Table() = default;

    template<class Archive>
    void serialize(Archive &ar)
    {
        ar(columns, types);
    }

    template <class R>
    std::vector<R> &getColumn(const size_t idx){
        return std::get<std::vector<R>>(columns.at(idx));
    }

  private:
    std::vector<column_type> columns;
    std::vector<ColumnTypes> types;
};

class CompressedTable : public Table
{
    std::vector<std::unordered_map<std::string, int32_t>> dictionaries;

  public:
    ~CompressedTable() override = default;
    void addColumn(column_type &&column, ColumnTypes type) override
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
};
#pragma once
#include "ColumnTypes.hpp"
#include "DateReformatter.hpp"
#include "DictionaryCompressor.hpp"

#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/variant.hpp>
#include <cereal/types/vector.hpp>
#include <string>
#include <utility>
#include <variant>
#include <vector>

class Table
{
  public:
    ColumnTypes getColType(size_t col);

    using column_type = std::variant<std::vector<std::string>, std::vector<double>, std::vector<int32_t>>;
    using row_type = std::variant<std::string, double, int32_t>;
    virtual void addColumn(column_type &&column, ColumnTypes type);

    virtual std::vector<row_type> getRow(size_t i);

    size_t columnCount();

    size_t rowCount();

    template<class R>
    std::vector<R> &getColumn(const size_t idx)
    {
        return std::get<std::vector<R>>(columns.at(idx));
    }

    virtual void addRow(std::vector<row_type> &row);

    virtual Table join(Table &other, size_t joinCol1, size_t joinCol2);

    static Table makeWideTable(std::vector<Table> tables, std::vector<std::pair<size_t, size_t>> joinCols);

    virtual ~Table() = default;
    Table() = default;
    Table(std::vector<column_type> columns, std::vector<ColumnTypes> types) :
        columns(std::move(columns)), types(std::move(types)){};

  protected:
    std::vector<column_type> columns;
    std::vector<ColumnTypes> types;
};

class CompressedTable : public Table
{
    std::vector<std::unordered_map<std::string, int32_t>> dictionaries;

  public:
    ~CompressedTable() override = default;
    CompressedTable() = default;
    CompressedTable(std::vector<column_type> columns,
                    std::vector<ColumnTypes> types,
                    std::vector<std::unordered_map<std::string, int32_t>> dictionaries) :
        Table(std::move(columns), std::move(types)), dictionaries(std::move(dictionaries)){};

    explicit CompressedTable(const std::string &path);

    CompressedTable(CompressedTable &) = default;
    explicit CompressedTable(Table &&tbl);

    void addColumn(column_type &&column, ColumnTypes type) final;

    template<class Archive>
    void serialize(Archive &ar)
    {
        ar(columns, types, dictionaries);
    }

    std::unordered_map<std::string, int32_t> getDictionary(size_t column);

    void writeTable(const std::string &path);
};
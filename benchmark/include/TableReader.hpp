#pragma once

#include "ColumnTypes.hpp"
#include "Tables.hpp"
#include "magic_enum.hpp"
#include "range/v3/view/enumerate.hpp"

#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_map.hpp>
#include <csv.hpp>
#include <cxxopts.hpp>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

class TableReader
{
    using CSVFormat = csv::CSVFormat;

  protected:
    using CSVReader = csv::CSVReader;
    CSVFormat format;
    std::string inFile;

    virtual std::vector<Table::column_type> getCols()
    {
        std::vector<Table::column_type> cols;
        cols.reserve(colTypes.size());
        for (auto &colType : colTypes)
        {
            switch (colType)
            {
                case ColumnTypes::INT:
                    cols.emplace_back(std::vector<int32_t>());
                    break;
                case ColumnTypes::STRING:
                    cols.emplace_back(std::vector<std::string>());
                    break;
                case ColumnTypes::DATE:
                    cols.emplace_back(std::vector<std::string>());
                    break;
                case ColumnTypes::DECIMAL:
                    cols.emplace_back(std::vector<double>());
                    break;
            }
        }
        return cols;
    }

    virtual void readTable(std::vector<Table::column_type> &cols)
    {
        CSVReader doc(inFile, format);
        for (auto &row : doc)
        {
            assert(row.size() == cols.size());
            for (size_t i = 0; i < colTypes.size(); ++i)
            {
                switch (colTypes.at(i))
                {
                    case ColumnTypes::INT:
                        std::get<std::vector<int32_t>>(cols.at(i)).push_back(row[i].get<int32_t>());
                        break;
                    case ColumnTypes::STRING:
                        std::get<std::vector<std::string>>(cols.at(i)).push_back(row[i].get<>());
                        break;
                    case ColumnTypes::DATE:
                        std::get<std::vector<std::string>>(cols.at(i)).push_back(row[i].get<>());
                        break;
                    case ColumnTypes::DECIMAL:
                        std::get<std::vector<double>>(cols.at(i)).push_back(row[i].get<double>());
                        break;
                }
            }
        }
    }
    std::vector<ColumnTypes> colTypes;

  public:
    TableReader(std::string inFile, std::vector<ColumnTypes> colTypes, char separator) :
        format(CSVFormat().delimiter(separator).no_header()), inFile(std::move(inFile)), colTypes(std::move(colTypes))
    {
    }

    virtual ~TableReader() = default;

    Table getTable()
    {
        std::vector<Table::column_type> cols = getCols();

        readTable(cols);

        return {std::move(cols), colTypes};
    }
};

class CompressingTableReader : TableReader
{
    using TableReader::TableReader;

    void readTable(std::vector<Table::column_type> &cols) override
    {
        std::vector<std::unordered_map<std::string, int32_t>> dictionaries(cols.size());
        {
            std::vector<std::set<std::string>> colVals(cols.size());
            CSVReader doc(inFile, format);
            for (const auto &row : doc)
            {
                for (size_t i = 0; i < row.size(); ++i)
                {
                    if (colTypes.at(i) == ColumnTypes::STRING)
                    {
                        colVals.at(i).emplace(row[i].get<>());
                    }
                }
            }

            for (const auto &s_d : ranges::views::zip(colVals, dictionaries))
            {
                std::set<std::string> s;
                std::unordered_map<std::string, int32_t> d;
                std::tie(s, d) = s_d;
                for (const auto &e : s)
                    d.emplace(e, d.size());
            }
        }
        CSVReader doc(inFile, format);
        for (const auto &row : doc)
        {
            for (size_t i = 0; i < row.size(); ++i)
            {
                switch (colTypes.at(i))
                {
                    case ColumnTypes::INT:
                        std::get<std::vector<int32_t>>(cols.at(i)).push_back(row[i].get<int32_t>());
                        break;
                    case ColumnTypes::STRING:
                        std::get<std::vector<int32_t>>(cols.at(i)).push_back(dictionaries[i][row[i].get<std::string>()]);
                        break;
                    case ColumnTypes::DATE:
                        std::get<std::vector<int32_t>>(cols.at(i)).push_back(DateReformatter::parseDate(row[i].get<std::string>()));
                        break;
                    case ColumnTypes::DECIMAL:
                        std::get<std::vector<double>>(cols.at(i)).push_back(row[i].get<double>());
                        break;
                }
            }
        }
    }

    std::vector<Table::column_type> getCols() override
    {
        std::vector<Table::column_type> cols;
        cols.reserve(colTypes.size());
        for (auto &colType : colTypes)
        {
            switch (colType)
            {
                case ColumnTypes::INT:
                    cols.emplace_back(std::vector<int32_t>());
                    break;
                case ColumnTypes::STRING:
                    cols.emplace_back(std::vector<int32_t>());
                    break;
                case ColumnTypes::DATE:
                    cols.emplace_back(std::vector<int32_t>());
                    break;
                case ColumnTypes::DECIMAL:
                    cols.emplace_back(std::vector<double>());
                    break;
            }
        }
        return cols;
    }

  public:
    CompressedTable getTable()
    {
        std::vector<Table::column_type> cols = getCols();
        size_t nRows = 0;
        std::vector<std::unordered_map<std::string, int32_t>> dictionaries(colTypes.size());
        {
            std::vector<std::set<std::string>> colVals(colTypes.size());
            CSVReader doc(inFile, format);
            for (auto &row : doc)
            {
                ++nRows;
                assert(colTypes.size() == row.size());
                for (size_t i = 0; i < row.size(); ++i)
                {
                    if (colTypes.at(i) == ColumnTypes::STRING)
                    {
                        colVals.at(i).emplace(row[i].get<std::string>());
                    }
                }
            }

            for (const auto &s_d : ranges::views::zip(colVals, dictionaries))
            {
                std::set<std::string> s;
                std::unordered_map<std::string, int32_t> d;
                std::tie(s, d) = s_d;
                for (const auto &e : s)
                    d.emplace(e, d.size());
            }
        }
        for (size_t i = 0; i < cols.size(); ++i)
        {
            switch (colTypes.at(i))
            {
                case ColumnTypes::INT:
                    std::get<std::vector<int32_t>>(cols.at(i)).reserve(nRows);
                    break;
                case ColumnTypes::STRING:
                    std::get<std::vector<int32_t>>(cols.at(i)).reserve(nRows);
                    break;
                case ColumnTypes::DATE:
                    std::get<std::vector<int32_t>>(cols.at(i)).reserve(nRows);
                    break;
                case ColumnTypes::DECIMAL:
                    std::get<std::vector<double>>(cols.at(i)).reserve(nRows);
                    break;
            }

        }
        CSVReader doc(inFile, format);
        for (const auto &row : doc)
        {
            for (size_t i = 0; i < row.size(); ++i)
            {
                switch (colTypes.at(i))
                {
                    case ColumnTypes::INT:
                        std::get<std::vector<int32_t>>(cols.at(i)).push_back(row[i].get<int32_t>());
                        break;
                    case ColumnTypes::STRING:
                        std::get<std::vector<int32_t>>(cols.at(i)).push_back(dictionaries[i][row[i].get<std::string>()]);
                        break;
                    case ColumnTypes::DATE:
                        std::get<std::vector<int32_t>>(cols.at(i)).push_back(DateReformatter::parseDate(row[i].get<std::string>()));
                        break;
                    case ColumnTypes::DECIMAL:
                        std::get<std::vector<double>>(cols.at(i)).push_back(row[i].get<double>());
                        break;
                }
            }
        }

        return {std::move(cols), colTypes, dictionaries};
    }
};
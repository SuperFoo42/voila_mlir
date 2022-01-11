#pragma once

#include "ColumnTypes.hpp"
#include "Tables.hpp"
#include "magic_enum.hpp"

#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cxxopts.hpp>
#include <iostream>

#include <rapidcsv.h>
#include <string>
#include <unordered_map>
#include <vector>

template<class TABLE_TYPE>
class TableReader
{
    std::vector<ColumnTypes> colTypes;
    rapidcsv::Document doc;

  public:
    TableReader(const std::string &inFile, std::vector<ColumnTypes> colTypes, char separator) :
        colTypes(std::move(colTypes)), doc(inFile, rapidcsv::LabelParams(), rapidcsv::SeparatorParams(separator))
    {
    }

    TABLE_TYPE getTable()
    {
        TABLE_TYPE tbl;
        for (size_t i = 0; i < colTypes.size(); ++i)
        {
            switch (colTypes[i])
            {
                case ColumnTypes::INT:
                    tbl.addColumn(doc.GetColumn<int32_t>(i), colTypes[i]);
                    break;
                case ColumnTypes::STRING:
                    tbl.addColumn(doc.GetColumn<std::string>(i), colTypes[i]);
                    break;
                case ColumnTypes::DATE:
                    tbl.addColumn(doc.GetColumn<std::string>(i), colTypes[i]);
                    break;
                case ColumnTypes::DECIMAL:
                    tbl.addColumn(doc.GetColumn<double>(i), colTypes[i]);
                    break;
            }
        }

        return tbl;
    }
};
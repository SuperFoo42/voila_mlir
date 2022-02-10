#pragma once
#include "Tables.hpp"

#include <memory>
#include <utility>

class BenchmarkState
{
    std::string dataPath;
    std::unique_ptr<CompressedTable> lineitem = nullptr;
    std::unique_ptr<CompressedTable> orders = nullptr;
    std::unique_ptr<CompressedTable> customer = nullptr;
    std::unique_ptr<CompressedTable> customer_orders_lineitem = nullptr;
    std::unique_ptr<CompressedTable> part_supplier_lineitem_partsupp_orders_nation = nullptr;

  public:
    explicit BenchmarkState(std::string datasetBase) :
        dataPath(std::move(datasetBase)),
        lineitem(nullptr),
        orders(nullptr),
        customer(nullptr),
        customer_orders_lineitem(nullptr),
        part_supplier_lineitem_partsupp_orders_nation(nullptr){};

    CompressedTable &getLineitemCompressed();

    CompressedTable &getOrdersCompressed();

    CompressedTable &getCustomerCompressed();

    CompressedTable &getCustomerOrderLineitem();

    CompressedTable &getPartSupplierLineitemPartsuppOrdersNation();
};
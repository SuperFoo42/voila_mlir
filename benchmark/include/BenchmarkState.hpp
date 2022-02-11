#pragma once
#include "Tables.hpp"

#include <memory>
#include <utility>

class BenchmarkState
{
    std::string dataPath;
    std::unique_ptr<CompressedTable> compressedLineitem = nullptr;
    std::unique_ptr<CompressedTable> compressedPart = nullptr;
    std::unique_ptr<CompressedTable> compressedSupplier = nullptr;
    std::unique_ptr<CompressedTable> compressedNation = nullptr;
    std::unique_ptr<CompressedTable> compressedPartsupp = nullptr;
    std::unique_ptr<CompressedTable> compressedOrders = nullptr;
    std::unique_ptr<CompressedTable> compressedCustomer = nullptr;
    std::unique_ptr<Table> lineitem = nullptr;
    std::unique_ptr<Table> part = nullptr;
    std::unique_ptr<Table> supplier = nullptr;
    std::unique_ptr<Table> nation = nullptr;
    std::unique_ptr<Table> partsupp = nullptr;
    std::unique_ptr<Table> orders = nullptr;
    std::unique_ptr<Table> customer = nullptr;
    std::unique_ptr<CompressedTable> customer_orders_lineitem = nullptr;
    std::unique_ptr<CompressedTable> part_supplier_lineitem_partsupp_orders_nation = nullptr;

  public:
    explicit BenchmarkState(std::string datasetBase) :
        dataPath(std::move(datasetBase)),
        compressedLineitem(nullptr),
        compressedPart(nullptr),
        compressedSupplier(nullptr),
        compressedNation(nullptr),
        compressedPartsupp(nullptr),
        compressedOrders(nullptr),
        compressedCustomer(nullptr),
        customer_orders_lineitem(nullptr),
        part_supplier_lineitem_partsupp_orders_nation(nullptr){};

    CompressedTable &getLineitemCompressed();

    CompressedTable &getOrdersCompressed();

    CompressedTable &getCustomerCompressed();

    CompressedTable &getCustomerOrderLineitem();

    CompressedTable &getPartSupplierLineitemPartsuppOrdersNation();

    CompressedTable &getPartCompressed();

    CompressedTable &getSupplierCompressed();

    CompressedTable &getPartsuppCompressed();

    CompressedTable &getNationCompressed();

    Table &getLineitem();

    Table &getOrders();

    Table &getCustomer();

    Table &getPart();

    Table &getSupplier();

    Table &getPartsupp();

    Table &getNation();

    void clear()
    {
        compressedLineitem.reset(nullptr);
        compressedPart.reset(nullptr);
        compressedSupplier.reset(nullptr);
        compressedPartsupp.reset(nullptr);
        compressedNation.reset(nullptr);
        compressedOrders.reset(nullptr);
        compressedCustomer.reset(nullptr);
        lineitem.reset(nullptr);
        part.reset(nullptr);
        supplier.reset(nullptr);
        nation.reset(nullptr);
        partsupp.reset(nullptr);
        orders.reset(nullptr);
        customer.reset(nullptr);
        customer_orders_lineitem.reset(nullptr);
        part_supplier_lineitem_partsupp_orders_nation.reset(nullptr);
    }
};
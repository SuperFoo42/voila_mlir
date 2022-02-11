#include "BenchmarkState.hpp"

#include "TableReader.hpp"

CompressedTable &BenchmarkState::getPartSupplierLineitemPartsuppOrdersNation()
{
    if (lineitem)
        lineitem.reset(nullptr);
    if (nation)
        nation.reset(nullptr);
    if (customer)
        customer.reset(nullptr);
    if (orders)
        orders.reset(nullptr);
    if (part)
        part.reset(nullptr);
    if (partsupp)
        partsupp.reset(nullptr);
    if (supplier)
        supplier.reset(nullptr);
    if (compressedLineitem)
        compressedLineitem.reset(nullptr);
    if (compressedOrders)
        compressedOrders.reset(nullptr);
    if (compressedCustomer)
        compressedCustomer.reset(nullptr);
    if (compressedPart)
        compressedPart.reset(nullptr);
    if (compressedNation)
        compressedNation.reset(nullptr);
    if (compressedPartsupp)
        compressedPartsupp.reset(nullptr);
    if (compressedSupplier)
        compressedSupplier.reset(nullptr);
    if (customer_orders_lineitem)
        customer_orders_lineitem.reset(nullptr);

    if (!part_supplier_lineitem_partsupp_orders_nation)
    {
        part_supplier_lineitem_partsupp_orders_nation =
            std::make_unique<CompressedTable>(dataPath + "/part_supplier_lineitem_partsupp_orders_nation.bin.xz");
    }
    return *part_supplier_lineitem_partsupp_orders_nation;
}
CompressedTable &BenchmarkState::getCustomerOrderLineitem()
{
    if (lineitem)
        lineitem.reset(nullptr);
    if (nation)
        nation.reset(nullptr);
    if (customer)
        customer.reset(nullptr);
    if (orders)
        orders.reset(nullptr);
    if (part)
        part.reset(nullptr);
    if (partsupp)
        partsupp.reset(nullptr);
    if (supplier)
        supplier.reset(nullptr);
    if (compressedLineitem)
        compressedLineitem.reset(nullptr);
    if (compressedOrders)
        compressedOrders.reset(nullptr);
    if (compressedCustomer)
        compressedCustomer.reset(nullptr);
    if (compressedPart)
        compressedPart.reset(nullptr);
    if (compressedNation)
        compressedNation.reset(nullptr);
    if (compressedPartsupp)
        compressedPartsupp.reset(nullptr);
    if (compressedSupplier)
        compressedSupplier.reset(nullptr);
    if (part_supplier_lineitem_partsupp_orders_nation)
        part_supplier_lineitem_partsupp_orders_nation.reset(nullptr);

    if (!customer_orders_lineitem)
        customer_orders_lineitem = std::make_unique<CompressedTable>(dataPath + "/customer_orders_lineitem.bin.xz");
    return *customer_orders_lineitem;
}
CompressedTable &BenchmarkState::getOrdersCompressed()
{
    if (lineitem)
        lineitem.reset(nullptr);
    if (nation)
        nation.reset(nullptr);
    if (customer)
        customer.reset(nullptr);
    if (orders)
        orders.reset(nullptr);
    if (part)
        part.reset(nullptr);
    if (partsupp)
        partsupp.reset(nullptr);
    if (supplier)
        supplier.reset(nullptr);
    if (customer_orders_lineitem)
        customer_orders_lineitem.reset(nullptr);
    if (part_supplier_lineitem_partsupp_orders_nation)
        part_supplier_lineitem_partsupp_orders_nation.reset(nullptr);

    if (!compressedOrders)
        compressedOrders = std::make_unique<CompressedTable>(dataPath + "/orders.bin.xz");
    return *compressedOrders;
}
CompressedTable &BenchmarkState::getCustomerCompressed()
{
    if (lineitem)
        lineitem.reset(nullptr);
    if (nation)
        nation.reset(nullptr);
    if (customer)
        customer.reset(nullptr);
    if (orders)
        orders.reset(nullptr);
    if (part)
        part.reset(nullptr);
    if (partsupp)
        partsupp.reset(nullptr);
    if (supplier)
        supplier.reset(nullptr);
    if (customer_orders_lineitem)
        customer_orders_lineitem.reset(nullptr);
    if (part_supplier_lineitem_partsupp_orders_nation)
        part_supplier_lineitem_partsupp_orders_nation.reset(nullptr);

    if (!compressedCustomer)
        compressedCustomer = std::make_unique<CompressedTable>(dataPath + "/customer.bin.xz");
    return *compressedCustomer;
}
CompressedTable &BenchmarkState::getLineitemCompressed()
{
    if (lineitem)
        lineitem.reset(nullptr);
    if (nation)
        nation.reset(nullptr);
    if (customer)
        customer.reset(nullptr);
    if (orders)
        orders.reset(nullptr);
    if (part)
        part.reset(nullptr);
    if (partsupp)
        partsupp.reset(nullptr);
    if (supplier)
        supplier.reset(nullptr);
    if (customer_orders_lineitem)
        customer_orders_lineitem.reset(nullptr);
    if (part_supplier_lineitem_partsupp_orders_nation)
        part_supplier_lineitem_partsupp_orders_nation.reset(nullptr);

    if (!compressedLineitem)
        compressedLineitem = std::make_unique<CompressedTable>(dataPath + "/lineitem.bin.xz");
    return *compressedLineitem;
}
CompressedTable &BenchmarkState::getPartCompressed()
{
    if (lineitem)
        lineitem.reset(nullptr);
    if (nation)
        nation.reset(nullptr);
    if (customer)
        customer.reset(nullptr);
    if (orders)
        orders.reset(nullptr);
    if (part)
        part.reset(nullptr);
    if (partsupp)
        partsupp.reset(nullptr);
    if (supplier)
        supplier.reset(nullptr);
    if (customer_orders_lineitem)
        customer_orders_lineitem.reset(nullptr);
    if (part_supplier_lineitem_partsupp_orders_nation)
        part_supplier_lineitem_partsupp_orders_nation.reset(nullptr);

    if (!compressedPart)
        compressedPart = std::make_unique<CompressedTable>(dataPath + "/part.bin.xz");
    return *compressedPart;
}
CompressedTable &BenchmarkState::getSupplierCompressed()
{
    if (lineitem)
        lineitem.reset(nullptr);
    if (nation)
        nation.reset(nullptr);
    if (customer)
        customer.reset(nullptr);
    if (orders)
        orders.reset(nullptr);
    if (part)
        part.reset(nullptr);
    if (partsupp)
        partsupp.reset(nullptr);
    if (supplier)
        supplier.reset(nullptr);
    if (customer_orders_lineitem)
        customer_orders_lineitem.reset(nullptr);
    if (part_supplier_lineitem_partsupp_orders_nation)
        part_supplier_lineitem_partsupp_orders_nation.reset(nullptr);

    if (!compressedSupplier)
        compressedSupplier = std::make_unique<CompressedTable>(dataPath + "/supplier.bin.xz");
    return *compressedSupplier;
}
CompressedTable &BenchmarkState::getPartsuppCompressed()
{
    if (lineitem)
        lineitem.reset(nullptr);
    if (nation)
        nation.reset(nullptr);
    if (customer)
        customer.reset(nullptr);
    if (orders)
        orders.reset(nullptr);
    if (part)
        part.reset(nullptr);
    if (partsupp)
        partsupp.reset(nullptr);
    if (supplier)
        supplier.reset(nullptr);
    if (customer_orders_lineitem)
        customer_orders_lineitem.reset(nullptr);
    if (part_supplier_lineitem_partsupp_orders_nation)
        part_supplier_lineitem_partsupp_orders_nation.reset(nullptr);

    if (!compressedPartsupp)
        compressedPartsupp = std::make_unique<CompressedTable>(dataPath + "/partsupp.bin.xz");
    return *compressedPartsupp;
}
CompressedTable &BenchmarkState::getNationCompressed()
{
    if (lineitem)
        lineitem.reset(nullptr);
    if (nation)
        nation.reset(nullptr);
    if (customer)
        customer.reset(nullptr);
    if (orders)
        orders.reset(nullptr);
    if (part)
        part.reset(nullptr);
    if (partsupp)
        partsupp.reset(nullptr);
    if (supplier)
        supplier.reset(nullptr);
    if (customer_orders_lineitem)
        customer_orders_lineitem.reset(nullptr);
    if (part_supplier_lineitem_partsupp_orders_nation)
        part_supplier_lineitem_partsupp_orders_nation.reset(nullptr);

    if (!compressedNation)
        compressedNation = std::make_unique<CompressedTable>(dataPath + "/nation.bin.xz");
    return *compressedNation;
}

Table &BenchmarkState::getNation()
{
    if (compressedLineitem)
        compressedLineitem.reset(nullptr);
    if (compressedNation)
        compressedNation.reset(nullptr);
    if (compressedCustomer)
        compressedCustomer.reset(nullptr);
    if (compressedOrders)
        compressedOrders.reset(nullptr);
    if (compressedPart)
        compressedPart.reset(nullptr);
    if (compressedPartsupp)
        compressedPartsupp.reset(nullptr);
    if (compressedSupplier)
        compressedSupplier.reset(nullptr);
    if (customer_orders_lineitem)
        customer_orders_lineitem.reset(nullptr);
    if (part_supplier_lineitem_partsupp_orders_nation)
        part_supplier_lineitem_partsupp_orders_nation.reset(nullptr);

    if (!nation)
        nation = std::make_unique<Table>(
            TableReader(dataPath + "/nation.tbl",
                        {ColumnTypes::INT, ColumnTypes::STRING, ColumnTypes::INT, ColumnTypes::STRING}, '|')
                .getTable());
    return *nation;
}

Table &BenchmarkState::getLineitem()
{
    if (compressedLineitem)
        compressedLineitem.reset(nullptr);
    if (compressedNation)
        compressedNation.reset(nullptr);
    if (compressedCustomer)
        compressedCustomer.reset(nullptr);
    if (compressedOrders)
        compressedOrders.reset(nullptr);
    if (compressedPart)
        compressedPart.reset(nullptr);
    if (compressedPartsupp)
        compressedPartsupp.reset(nullptr);
    if (compressedSupplier)
        compressedSupplier.reset(nullptr);
    if (customer_orders_lineitem)
        customer_orders_lineitem.reset(nullptr);
    if (part_supplier_lineitem_partsupp_orders_nation)
        part_supplier_lineitem_partsupp_orders_nation.reset(nullptr);

    if (!lineitem)
        lineitem = std::make_unique<Table>(
            TableReader(dataPath + "/lineitem.tbl",
                        {ColumnTypes::INT, ColumnTypes::INT, ColumnTypes::INT, ColumnTypes::INT, ColumnTypes::DECIMAL,
                         ColumnTypes::DECIMAL, ColumnTypes::DECIMAL, ColumnTypes::DECIMAL, ColumnTypes::STRING,
                         ColumnTypes::STRING, ColumnTypes::DATE, ColumnTypes::DATE, ColumnTypes::DATE,
                         ColumnTypes::STRING, ColumnTypes::STRING, ColumnTypes::STRING},
                        '|')
                .getTable());
    return *lineitem;
}

Table &BenchmarkState::getOrders()
{
    if (compressedLineitem)
        compressedLineitem.reset(nullptr);
    if (compressedNation)
        compressedNation.reset(nullptr);
    if (compressedCustomer)
        compressedCustomer.reset(nullptr);
    if (compressedOrders)
        compressedOrders.reset(nullptr);
    if (compressedPart)
        compressedPart.reset(nullptr);
    if (compressedPartsupp)
        compressedPartsupp.reset(nullptr);
    if (compressedSupplier)
        compressedSupplier.reset(nullptr);
    if (customer_orders_lineitem)
        customer_orders_lineitem.reset(nullptr);
    if (part_supplier_lineitem_partsupp_orders_nation)
        part_supplier_lineitem_partsupp_orders_nation.reset(nullptr);

    if (!orders)
        orders = std::make_unique<Table>(TableReader(dataPath + "/orders.tbl",
                                                     {ColumnTypes::INT, ColumnTypes::INT, ColumnTypes::STRING,
                                                      ColumnTypes::DECIMAL, ColumnTypes::DATE, ColumnTypes::STRING,
                                                      ColumnTypes::STRING, ColumnTypes::INT, ColumnTypes::STRING},
                                                     '|')
                                             .getTable());
    return *orders;
}

Table &BenchmarkState::getCustomer()
{
    if (compressedLineitem)
        compressedLineitem.reset(nullptr);
    if (compressedNation)
        compressedNation.reset(nullptr);
    if (compressedCustomer)
        compressedCustomer.reset(nullptr);
    if (compressedOrders)
        compressedOrders.reset(nullptr);
    if (compressedPart)
        compressedPart.reset(nullptr);
    if (compressedPartsupp)
        compressedPartsupp.reset(nullptr);
    if (compressedSupplier)
        compressedSupplier.reset(nullptr);
    if (customer_orders_lineitem)
        customer_orders_lineitem.reset(nullptr);
    if (part_supplier_lineitem_partsupp_orders_nation)
        part_supplier_lineitem_partsupp_orders_nation.reset(nullptr);

    if (!customer)
        customer = std::make_unique<Table>(
            TableReader(dataPath + "/customer.tbl",
                        {ColumnTypes::INT, ColumnTypes::STRING, ColumnTypes::STRING, ColumnTypes::INT,
                         ColumnTypes::STRING, ColumnTypes::DECIMAL, ColumnTypes::STRING, ColumnTypes::STRING},
                        '|')
                .getTable());
    return *customer;
}

Table &BenchmarkState::getPart()
{
    if (compressedLineitem)
        compressedLineitem.reset(nullptr);
    if (compressedNation)
        compressedNation.reset(nullptr);
    if (compressedCustomer)
        compressedCustomer.reset(nullptr);
    if (compressedOrders)
        compressedOrders.reset(nullptr);
    if (compressedPart)
        compressedPart.reset(nullptr);
    if (compressedPartsupp)
        compressedPartsupp.reset(nullptr);
    if (compressedSupplier)
        compressedSupplier.reset(nullptr);
    if (customer_orders_lineitem)
        customer_orders_lineitem.reset(nullptr);
    if (part_supplier_lineitem_partsupp_orders_nation)
        part_supplier_lineitem_partsupp_orders_nation.reset(nullptr);

    if (!part)
        part = std::make_unique<Table>(TableReader(dataPath + "/part.tbl",
                                                   {ColumnTypes::INT, ColumnTypes::STRING, ColumnTypes::STRING,
                                                    ColumnTypes::STRING, ColumnTypes::STRING, ColumnTypes::INT,
                                                    ColumnTypes::STRING, ColumnTypes::DECIMAL, ColumnTypes::STRING},
                                                   '|')
                                           .getTable());
    return *part;
}

Table &BenchmarkState::getPartsupp()
{
    if (compressedLineitem)
        compressedLineitem.reset(nullptr);
    if (compressedNation)
        compressedNation.reset(nullptr);
    if (compressedCustomer)
        compressedCustomer.reset(nullptr);
    if (compressedOrders)
        compressedOrders.reset(nullptr);
    if (compressedPart)
        compressedPart.reset(nullptr);
    if (compressedPartsupp)
        compressedPartsupp.reset(nullptr);
    if (compressedSupplier)
        compressedSupplier.reset(nullptr);
    if (customer_orders_lineitem)
        customer_orders_lineitem.reset(nullptr);
    if (part_supplier_lineitem_partsupp_orders_nation)
        part_supplier_lineitem_partsupp_orders_nation.reset(nullptr);

    if (!partsupp)
        partsupp = std::make_unique<Table>(TableReader(dataPath + "/partsupp.tbl",
                                                       {ColumnTypes::INT, ColumnTypes::INT, ColumnTypes::INT,
                                                        ColumnTypes::DECIMAL, ColumnTypes::STRING},
                                                       '|')
                                               .getTable());
    return *partsupp;
}

Table &BenchmarkState::getSupplier()
{
    if (compressedLineitem)
        compressedLineitem.reset(nullptr);
    if (compressedNation)
        compressedNation.reset(nullptr);
    if (compressedCustomer)
        compressedCustomer.reset(nullptr);
    if (compressedOrders)
        compressedOrders.reset(nullptr);
    if (compressedPart)
        compressedPart.reset(nullptr);
    if (compressedPartsupp)
        compressedPartsupp.reset(nullptr);
    if (compressedSupplier)
        compressedSupplier.reset(nullptr);
    if (customer_orders_lineitem)
        customer_orders_lineitem.reset(nullptr);
    if (part_supplier_lineitem_partsupp_orders_nation)
        part_supplier_lineitem_partsupp_orders_nation.reset(nullptr);

    if (!supplier)
        supplier = std::make_unique<Table>(
            TableReader(dataPath + "/supplier.tbl",
                        {ColumnTypes::INT, ColumnTypes::STRING, ColumnTypes::STRING, ColumnTypes::INT,
                         ColumnTypes::STRING, ColumnTypes::DECIMAL, ColumnTypes::STRING},
                        '|')
                .getTable());
    return *supplier;
}
#include "BenchmarkState.hpp"

#include "TableReader.hpp"

CompressedTable &BenchmarkState::getPartSupplierLineitemPartsuppOrdersNation()
{
    if (lineitem)
        lineitem.reset(nullptr);
    if (orders)
        orders.reset(nullptr);
    if (customer)
        customer.reset(nullptr);
    if (customer_orders_lineitem)
        customer_orders_lineitem.reset(nullptr);

    if (!part_supplier_lineitem_partsupp_orders_nation)
    {
        part_supplier_lineitem_partsupp_orders_nation = std::make_unique<CompressedTable>(
            dataPath + "/part_supplier_lineitem_partsupp_orders_nation.bin.xz");
    }
    return *part_supplier_lineitem_partsupp_orders_nation;
}
CompressedTable &BenchmarkState::getCustomerOrderLineitem()
{
    if (lineitem)
        lineitem.reset(nullptr);
    if (orders)
        orders.reset(nullptr);
    if (customer)
        customer.reset(nullptr);
    if (part_supplier_lineitem_partsupp_orders_nation)
        part_supplier_lineitem_partsupp_orders_nation.reset(nullptr);

    if (!customer_orders_lineitem)
        customer_orders_lineitem =
            std::make_unique<CompressedTable>(dataPath + "/customer_orders_lineitem.bin.xz");
    return *customer_orders_lineitem;
}
CompressedTable &BenchmarkState::getOrdersCompressed()
{
    if (customer_orders_lineitem)
        customer_orders_lineitem.reset(nullptr);
    if (part_supplier_lineitem_partsupp_orders_nation)
        part_supplier_lineitem_partsupp_orders_nation.reset(nullptr);

    if (!orders)
        orders = std::make_unique<CompressedTable>(dataPath + "/orders.bin.xz");
    return *orders;
}
CompressedTable &BenchmarkState::getCustomerCompressed()
{
    if (customer_orders_lineitem)
        customer_orders_lineitem.reset(nullptr);
    if (part_supplier_lineitem_partsupp_orders_nation)
        part_supplier_lineitem_partsupp_orders_nation.reset(nullptr);

    if (!customer)
        customer = std::make_unique<CompressedTable>(dataPath + "/customer.bin.xz");
    return *customer;
}
CompressedTable &BenchmarkState::getLineitemCompressed()
{
    if (customer_orders_lineitem)
        customer_orders_lineitem.reset(nullptr);
    if (part_supplier_lineitem_partsupp_orders_nation)
        part_supplier_lineitem_partsupp_orders_nation.reset(nullptr);

    if (!lineitem)
        lineitem = std::make_unique<CompressedTable>(dataPath + "/lineitem.bin.xz");
    return *lineitem;
}

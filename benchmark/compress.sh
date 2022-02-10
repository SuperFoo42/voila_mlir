#!/usr/bin/env bash

#100g
#../cmake-build-release/benchmark/src/tools/preprocessor -t INT,INT,INT,INT,DECIMAL,DECIMAL,DECIMAL,DECIMAL,STRING,STRING,DATE,DATE,DATE,STRING,STRING,STRING -f /home/paul/Downloads/2.18.0_rc2/dbgen/lineitem.tbl -o /home/paul/Projekte/AdaptiveReprogramming/tpc-h/SF100/lineitem100g.bin.xz
../cmake-build-release/benchmark/src/tools/preprocessor -t INT,INT,STRING,DECIMAL,DATE,STRING,STRING,INT,STRING -f /home/paul/Projekte/AdaptiveReprogramming/tpc-h/SF100/orders.tbl -o /home/paul/Projekte/AdaptiveReprogramming/tpc-h/SF100/orders100g.bin.xz
../cmake-build-release/benchmark/src/tools/preprocessor -t INT,STRING,STRING,INT,STRING,DECIMAL,STRING,STRING -f /home/paul/Projekte/AdaptiveReprogramming/tpc-h/SF100/customer.tbl -o /home/paul/Projekte/AdaptiveReprogramming/tpc-h/SF100/customer100g.bin.xz
../cmake-build-release/benchmark/src/tools/preprocessor -t INT,STRING,STRING,STRING,STRING,INT,STRING,DECIMAL,STRING -f /home/paul/Projekte/AdaptiveReprogramming/tpc-h/SF100/part.tbl -o /home/paul/Projekte/AdaptiveReprogramming/tpc-h/SF100/part100g.bin.xz
../cmake-build-release/benchmark/src/tools/preprocessor -t INT,STRING,STRING,INT,STRING,DECIMAL,STRING -f /home/paul/Projekte/AdaptiveReprogramming/tpc-h/SF100/supplier.tbl -o /home/paul/Projekte/AdaptiveReprogramming/tpc-h/SF100/supplier100g.bin.xz
../cmake-build-release/benchmark/src/tools/preprocessor -t INT,INT,INT,DECIMAL,STRING -f /home/paul/Projekte/AdaptiveReprogramming/tpc-h/SF100/partsupp.tbl -o /home/paul/Projekte/AdaptiveReprogramming/tpc-h/SF100/partsupp100g.bin.xz
../cmake-build-release/benchmark/src/tools/preprocessor -t INT,STRING,INT,STRING -f /home/paul/Projekte/AdaptiveReprogramming/tpc-h/SF100/nation.tbl -o /home/paul/Projekte/AdaptiveReprogramming/tpc-h/SF100/nation100g.bin.xz

echo "finished compressing all tables"


#!/usr/bin/env python3

import argparse
import csv

from datetime import date
from multiprocessing import Pool


def to_int(dt_date: date):
    return 10000 * dt_date.year + 100 * dt_date.month + dt_date.day


def readDataFrame(reader: csv.reader, cols: list[str]):
    df = [[] for _ in range(len(cols))]
    for row in reader:
        for i, t in enumerate(cols):
            if t == "INT":
                df[i].append(int(row[i]))
            elif t == "FLOAT":
                df[i].append(float(row[i]))
            elif t == "DATE":
                # convert to int in iso
                df[i].append(to_int(date.fromisoformat(row[i])))
            else:  # to dictionary compress
                df[i].append(row[i])
    return df


def compressCol(col):
    d = list(set(col))
    d.sort()
    d = dict(zip(list(d), list(range(len(d)))))
    compressedCol = [d[elem] for elem in col]
    return d, compressedCol


def compressFrame(dataFrame, types):
    dicts = []
    res = []
    p = Pool(None)
    for i, type in enumerate(types):
        if type == "STRING":
            res.append(p.apply_async(compressCol, args=(dataFrame[i],)))
    p.close()
    p.join()
    for i, type in enumerate(types):
        if type == "STRING":
            r = res.pop(0).get()
            dicts.append(r[0])
            dataFrame[i] = r[1]
    return dicts, dataFrame


def compressTable(file: str, delim: str, types: list[str]):
    with open(file, newline='\n') as csvfile:
        reader = None
        if delim is None:
            dialect = csv.Sniffer().sniff(csvfile.read(2048))
            csvfile.seek(0)
            reader = csv.reader(csvfile, dialect)
        else:
            reader = csv.reader(csvfile, delimiter=delim)
        dataFrame = readDataFrame(reader, types)
        return compressFrame(dataFrame, types)


def writeTable(file: str, df):
    with open(file, 'w', newline='\n') as out:
        writer = csv.writer(out, delimiter='|', quoting=csv.QUOTE_NONE)
        for i in range(len(df[0])):
            writer.writerow([df[j][i] for j in range(len(df))])


def writeDict(d, col, outfile):
    with open("{}.{}.{}.{}".format("col", col,"dict",outfile), 'w', newline='\n') as out:
        writer = csv.writer(out, delimiter='|', quoting=csv.QUOTE_NONE)
        for elem in d:
            writer.writerow((elem, d[elem]))


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file',
                    dest='file',
                    help='CSV-file to read and dictionary compress',
                    type=str
                    )

parser.add_argument('-o', '--out',
                    dest='outfile',
                    help='File to write the compressed table in csv to',
                    type=str
                    )

parser.add_argument('type',
                    nargs='+',
                    choices=('INT', 'FLOAT', 'STRING', 'DATE'),
                    help='Provide column types of csv file'
                    )

parser.add_argument('-d', '--delim',
                    dest='delim',
                    type=str,
                    default=None,
                    help='Delimiter of CSV file'
                    )

args = parser.parse_args()

dicts, compressedTable = compressTable(args.file, args.delim, args.type)
writeTable(args.outfile, compressedTable)

for col, type in enumerate(args.type):
    if type == "STRING":
        writeDict(dicts.pop(0), col, args.outfile)

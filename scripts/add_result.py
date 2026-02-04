#!/usr/bin/env python
from typing import Any, Dict, List, Union
import click
import csv
from elasticsearch import Elasticsearch


Row = Dict[str, Union[str, float]]


def open_csv(filename: str) -> List[Dict[str, str]]:
    """
    Open a csv file a return it as dictionary.
    First row is titles.
    """
    csv_rows = []
    with open(filename) as csv_data:
        reader = csv.DictReader(csv_data)
        titles = reader.fieldnames
        for row in reader:
            csv_rows.append(
                {
                    title: row[title]
                    for title in titles
                }
            )
    return csv_rows

def format_entry_accuracy(row: Row) -> Dict[str, Any]:
    """"format a result"""
    commit_sha   = str(row.pop('gitcommit'))
    commit_date  = str(row.pop('gitcommitdate'))
    hostname     = str(row.pop('hostname'))
    ndim         = int(row.pop('ndim'))
    kernel_type  = int(row.pop('kernel_type'))
    interp_type  = int(row.pop('interp_type'))
    tree_height  = int(row.pop('tree_height'))
    interp_order = int(row.pop('interp_order'))
    error        = float(row.pop('error'))
    result = {
        "Commit_date": commit_date,
        "Commit_sha": commit_sha,
        "Hostname": hostname,
        "Ndim": ndim,
        "Kernel_type": kernel_type,
        "Interp_type": interp_type,
        "Tree_height": tree_height,
        "Interp_order": interp_order,
        "Error": error
    }
    return result

def format_entry_timeseq(row: Row) -> Dict[str, Any]:
    """"format a result"""
    commit_sha   = str(row.pop('gitcommit'))
    commit_date  = str(row.pop('gitcommitdate'))
    hostname     = str(row.pop('hostname'))
    ndim         = int(row.pop('ndim'))
    kernel_type  = int(row.pop('kernel_type'))
    interp_type  = int(row.pop('interp_type'))
    nrun         = int(row.pop('nrun'))
    tree_height  = int(row.pop('tree_height'))
    interp_order = int(row.pop('interp_order'))
    size         = int(row.pop('size'))
    timefar_avg  = float(row.pop('timefar_avg'))
    timenear_avg = float(row.pop('timenear_avg'))
    timefull_avg = float(row.pop('timefull_avg'))

    result = {
        "Commit_date": commit_date,
        "Commit_sha": commit_sha,
        "Hostname": hostname,
        "Ndim": ndim,
        "Kernel_type": kernel_type,
        "Interp_type": interp_type,
        "Nrun": nrun,
        "Tree_height": tree_height,
        "Interp_order": interp_order,
        "Size": size,
        "Timefar_avg": timefar_avg,
        "Timenear_avg": timenear_avg,
        "Timefull_avg": timefull_avg
    }
    return result

def format_entry_timeomp(row: Row) -> Dict[str, Any]:
    """"format a result"""
    commit_sha   = str(row.pop('gitcommit'))
    commit_date  = str(row.pop('gitcommitdate'))
    hostname     = str(row.pop('hostname'))
    ndim         = int(row.pop('ndim'))
    kernel_type  = int(row.pop('kernel_type'))
    interp_type  = int(row.pop('interp_type'))
    nrun         = int(row.pop('nrun'))
    tree_height  = int(row.pop('tree_height'))
    interp_order = int(row.pop('interp_order'))
    size         = int(row.pop('size'))
    nthread      = int(row.pop('nthread'))
    groupsize    = int(row.pop('groupsize'))
    timefull_avg = float(row.pop('timefull_avg'))

    result = {
        "Commit_date": commit_date,
        "Commit_sha": commit_sha,
        "Hostname": hostname,
        "Ndim": ndim,
        "Kernel_type": kernel_type,
        "Interp_type": interp_type,
        "Nrun": nrun,
        "Tree_height": tree_height,
        "Interp_order": interp_order,
        "Size": size,
        "Nthread": nthread,
        "Groupsize": groupsize,
        "Timefull_avg": timefull_avg
    }
    return result

@click.command()
@click.option("-e", "--elastic-url", default="http://localhost:9200", help="elasticsearch instance url")
@click.option("-t", "--team", required=True, help="team name")
@click.option("-p", "--project", required=True, help="project name")
@click.option("-n", "--name", required=True, help="Table name")
@click.argument("csv-files", nargs=-1)
def main(
    elastic_url: str,
    team: str,
    project: str,
    name: str,
    csv_files: str,
):
    """Add a result to an elasticsearch database."""
    es = Elasticsearch(elastic_url)
    info = es.info()
    print("Elasticsearch version:", info["version"]["number"])

    es_index = team + "-" + project + "-" + name
    if not es.indices.exists(index=es_index):
        es.indices.create(index=es_index)

    mapping_input_accuracy = {
        "properties": {
            "Commit_sha": {"type": "keyword"},
            "Commit_date": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss Z"},
            "Hostname": {"type": "keyword"},
            "Ndim": {"type": "integer"},
            "Kernel_type": {"type": "integer"},
            "Interp_type": {"type": "integer"},
            "Tree_height": {"type": "integer"},
            "Interp_order": {"type": "integer"},
            "Error": {"type": "float"}
        }
    }
    mapping_input_timeseq = {
        "properties": {
            "Commit_sha": {"type": "keyword"},
            "Commit_date": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss Z"},
            "Hostname": {"type": "keyword"},
            "Ndim": {"type": "integer"},
            "Kernel_type": {"type": "integer"},
            "Interp_type": {"type": "integer"},
            "Nrun": {"type": "integer"},
            "Tree_height": {"type": "integer"},
            "Interp_order": {"type": "integer"},
            "Size": {"type": "integer"},
            "Timefar_avg": {"type": "float"},
            "Timenear_avg": {"type": "float"},
            "Timefull_avg": {"type": "float"}
        }
    }
    mapping_input_timeomp = {
        "properties": {
            "Commit_sha": {"type": "keyword"},
            "Commit_date": {"type": "date", "format": "yyyy-MM-dd HH:mm:ss Z"},
            "Hostname": {"type": "keyword"},
            "Ndim": {"type": "integer"},
            "Kernel_type": {"type": "integer"},
            "Interp_type": {"type": "integer"},
            "Nrun": {"type": "integer"},
            "Tree_height": {"type": "integer"},
            "Interp_order": {"type": "integer"},
            "Size": {"type": "integer"},
            "Nthread": {"type": "integer"},
            "Groupsize": {"type": "integer"},
            "Timefull_avg": {"type": "float"}
        }
    }

    if name == "accuracy":
        es.indices.put_mapping(index=es_index, body=mapping_input_accuracy)
    elif name == "timeseq":
        es.indices.put_mapping(index=es_index, body=mapping_input_timeseq)
    elif name == "timeomp":
        es.indices.put_mapping(index=es_index, body=mapping_input_timeomp)

    requests = [
        request
        for file in csv_files
            for request in map(
                lambda row: format_entry_accuracy(row) if name == "accuracy" else format_entry_timeseq(row) if name == "timeseq" else format_entry_timeomp(row),
                open_csv(file)
            )
    ]
    for request in requests:
        es.index(index=es_index.lower(), body=request)


if __name__ == "__main__":
    main()

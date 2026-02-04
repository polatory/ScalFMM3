#!/usr/bin/env python
import pandas
import click
import csv
from elasticsearch import Elasticsearch

@click.command()
@click.option("-e", "--elastic-url", default="http://localhost:9200", help="elasticsearch instance url")
@click.option("-t", "--team", required=True, help="team name")
@click.option("-p", "--project", required=True, help="project name")
@click.option("-n", "--name", required=True, help="Table name")
@click.option("-c", "--commit", required=True, help="project commit")
def main(
    elastic_url: str,
    team: str,
    project: str,
    name: str,
    commit: str
):
    """Get a result from an elasticsearch database, e.g.
    https://elasticsearch.bordeaux.inria.fr."""
    es = Elasticsearch(elastic_url)
    es_index = team + "-" + project + "-" + name

    search_param = {
      "query": {
        "bool": {
          "must": [
            {"term": {"Commit_sha": {"value": commit}}}
          ]
        }
      },
      "size": 1000
    }
    response = es.search(index=es_index, body=search_param)
    elastic_docs = response["hits"]["hits"]

    docs = pandas.DataFrame()
    for num, doc in enumerate(elastic_docs):

        # get _source data dict from document
        source_data = doc["_source"]

        # get _id from document
        _id = doc["_id"]

        # create a Series object from doc dict object
        doc_data = pandas.Series(source_data, name = _id)
        doc_data = doc_data.drop(labels=['Commit_date', 'Commit_sha'])

        # append the Series object to the DataFrame object
        docs = pandas.concat([docs, doc_data.to_frame().T])

    if name == 'accuracy':
        docs = docs.astype({"Hostname": str, "Ndim": int, "Kernel_type": int, "Interp_type": int, "Tree_height": int, "Interp_order": int, "Error": float})
    elif name == 'timeseq':
        docs = docs.astype({"Hostname": str, "Ndim": int, "Kernel_type": int, "Interp_type": int, "Tree_height": int, "Interp_order": int, "Size": int, "Timefar_avg": float, "Timenear_avg": float, "Timefull_avg": float})
    elif name == 'timeomp':
        docs = docs.astype({"Hostname": str, "Ndim": int, "Kernel_type": int, "Interp_type": int, "Nrun": int, "Tree_height": int, "Interp_order": int, "Size": int, "Nthread": int, "Groupsize": int, "Timefull_avg": float})

    docs = docs.rename(columns=str.lower)
    docs.to_csv("scalfmm_" + name + ".csv", ",", index=False)

if __name__ == "__main__":
    main()

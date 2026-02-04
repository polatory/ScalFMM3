#!/usr/bin/env bash
set -ex

# get current database file scalfmm.sqlite3 stored on gitlab
if [[ ! -z "${CI_JOB_TOKEN}" ]]; then
  export PACKAGEID=`curl --header "JOB-TOKEN: $CI_JOB_TOKEN" "https://gitlab.inria.fr/api/v4/projects/$CI_PROJECT_ID/packages" |jq '.[0].id'`
  if [[ "${PACKAGEID}" != "null" ]]; then
    export FILEID=`curl --header "JOB-TOKEN: $CI_JOB_TOKEN" "https://gitlab.inria.fr/api/v4/projects/$CI_PROJECT_ID/packages/$PACKAGEID/package_files" |jq '.[0].id'`
    if [[ "${FILEID}" != "null" ]]; then
      curl --header "JOB-TOKEN: $CI_JOB_TOKEN" "https://gitlab.inria.fr/api/v4/projects/$CI_PROJECT_ID/packages/generic/benchmark/latest/scalfmm.sqlite3" -o scalfmm.sqlite3
    fi
  fi
fi

# update scalfmm.sqlite3 database, tables : accuracy, timeseq, timeomp
jube result scripts/results --id 1 -o accuracy
jube result scripts/results --id 2 -o timeseq
jube result scripts/results --id 3 -o timeomp

# upload updated scalfmm.sqlite3 on gitlab
if [[ ! -z "${CI_JOB_TOKEN}" ]]; then
  # get package id of the database stored on gitlab
  export PACKAGEID=`curl --header "JOB-TOKEN: $CI_JOB_TOKEN" "https://gitlab.inria.fr/api/v4/projects/$CI_PROJECT_ID/packages" |jq '.[0].id'`
  if [[ "${PACKAGEID}" != "null" ]]; then
    # get file id of the database on gitlab
    export FILEID=`curl --header "JOB-TOKEN: $CI_JOB_TOKEN" "https://gitlab.inria.fr/api/v4/projects/$CI_PROJECT_ID/packages/$PACKAGEID/package_files" |jq '.[0].id'`
    if [[ "${FILEID}" != "null" ]]; then
      # delete previous database version on gitlab if exists
      curl --request DELETE --header "JOB-TOKEN: $CI_JOB_TOKEN" "https://gitlab.inria.fr/api/v4/projects/$CI_PROJECT_ID/packages/$PACKAGEID/package_files/$FILEID"
    fi
  fi
  curl --header "JOB-TOKEN: $CI_JOB_TOKEN" --upload-file ./scalfmm.sqlite3 "https://gitlab.inria.fr/api/v4/projects/$CI_PROJECT_ID/packages/generic/benchmark/latest/scalfmm.sqlite3"
fi

# upload results to the elasticsearch server
python3 ./scripts/add_result.py -e https://elasticsearch.bordeaux.inria.fr -t concace -p scalfmm -n accuracy scalfmm_accuracy.csv
python3 ./scripts/add_result.py -e https://elasticsearch.bordeaux.inria.fr -t concace -p scalfmm -n timeseq scalfmm_timeseq.csv
python3 ./scripts/add_result.py -e https://elasticsearch.bordeaux.inria.fr -t concace -p scalfmm -n timeomp scalfmm_timeomp.csv

# plot some figures at this precise commit
python3 ./scripts/plot_wrong_accuracy.py -f scalfmm_accuracy.csv -n accuracy
python3 ./scripts/plot.py -f scalfmm_accuracy.csv -n accuracy
python3 ./scripts/plot.py -f scalfmm_timeseq.csv -n timeseq
python3 ./scripts/plot.py -f scalfmm_timeomp.csv -n timeomp

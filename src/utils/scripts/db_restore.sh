#!/bin/bash

RESTORE_UTILITY='/opt/mongodb/bin/mongorestore'
DB_NAME='my_db'

log() {
    echo $1
}


do_restore(){
    local fname=$1
    tar -jxf $1 && \
    ${RESTORE_UTILITY} -d ${DB_NAME} ${fname//.bz2}/${DB_NAME}
}

get_file_from_s3(){
    python aws_s3.py get $1
}

do_cleanup(){
    rm -rf db_backup_2010*
    log 'cleaning up....'
}

if [ $# -lt 1 ]
then
    echo "Usage: $0 <backup_filename>"
    exit 1
fi

filename=$1

get_file_from_s3 ${filename} && do_restore ${filename} && do_cleanup
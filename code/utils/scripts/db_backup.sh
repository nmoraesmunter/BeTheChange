#!/bin/sh

MONGODB_SHELL='/usr/local/bin/mongo'

DUMP_UTILITY='/usr/local/bin/mongodump'
DB_NAME='copy_changeorg'

date_now=`date +%Y_%m_%d_%H_%M_%S`
dir_name='db_backup_'${date_now}
file_name='db_backup_'${date_now}'.bz2'
file_name='db_backup_2016_07_23_11_24_29.bz2'

log() {
    echo $1
}

do_cleanup(){
    rm -rf db_backup_2010*
    log 'cleaning up....'
}

do_backup(){
    log 'snapshotting the db and creating archive' && \
    ${MONGODB_SHELL} admin fsync_lock.js && \
    ${DUMP_UTILITY} -d ${DB_NAME} -o ${dir_name} && tar -jcf $file_name ${dir_name}
    ${MONGODB_SHELL} admin unlock.js && \
    log 'data backd up and created snapshot'
}

save_in_s3(){
    log 'saving the backup archive in amazon S3' && \
    python aws_s3.py set ${file_name} && \
    log 'data backup saved in amazon s3'
}

save_in_s3
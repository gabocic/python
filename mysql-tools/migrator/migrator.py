#!/usr/bin/python

import MySQLdb
import csv
import sys

hostlist='hostlist.csv'
dbuser = "percona"
dbpass = "percona"

def main():
    def save_slaves_metadata(name,cur_is_slave):
        sls_out = cur_is_slave.fetchone()
        print(sls_out[9])
        print(sls_out[21])
        slave_relay_masterbinlog[name] = sls_out[9]


    try:
        with open(hostlist,'rb') as csvfile:
            reader = csv.DictReader(csvfile)
            slave_relay_masterbinlog={}

            # For each row in the CSV
            for dbparams in reader:
                try:
                    # Attempt a MySQL connection
                    db = MySQLdb.connect(host=dbparams['ip'],port=int(dbparams['port']),user=dbuser,passwd=dbpass,db="information_schema")
                except:
                    print(sys.exc_info())
                    print("Connection to the database failed!")
                else:
                    has_slaves = 0
                    is_slave = 0
                    cursor = db.cursor()
                    cur_slave_hosts = db.cursor()
                    cur_is_slave = db.cursor()
                    cur_slave_hosts.execute("show slave hosts")
                    cur_is_slave.execute("show slave status")

                    # Check if "show slave hosts" returned at least 1 row
                    if cur_slave_hosts.rowcount > 0:
                        has_slaves = 1
                        for data in cur_slave_hosts:
                            print(data)

                    # Check if "show slave status" returned at least 1 row
                    if cur_is_slave.rowcount > 0:
                        is_slave = 1

                    # Infer node role based on the queries above
                    if has_slaves == 1 and is_slave == 0:
                        print("Host "+dbparams['name']+" seems to be master")
                        master_cur_binlog = cursor.execute("show master status")
                        print(cursor.fetchone())
                    elif has_slaves == 1 and is_slave == 1:
                        print("Host "+dbparams['name']+" seems to be an intermediary node")
                        save_slaves_metadata(dbparams['name'],cur_is_slave)
                    elif has_slaves == 0 and is_slave == 1:
                        print("Host "+dbparams['name']+" seems to be a slave")
                        save_slaves_metadata(dbparams['name'],cur_is_slave)
                    db.close()


    except:
        print("Exception opening hostlist files")
    else:
        print(slave_relay_masterbinlog)

if __name__ == '__main__':
    main()


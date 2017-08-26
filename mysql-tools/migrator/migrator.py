#!/usr/bin/python

import MySQLdb
import csv
import sys

hostlist='hostlist.csv'
dbuser = "percona"
dbpass = "percona"


## <<< Flush binlog at the master at the beginning to have 1Gb of time to perform the changes

def load_slaves():

    hosts_dict={}
    try:
        # Open csv containing mysql hosts data
        with open(hostlist,'rb') as csvfile:
            reader = csv.DictReader(csvfile)

            # For each row in the CSV
            for dbparams in reader:
                try:
                    # Attempt a MySQL connection
                    db = MySQLdb.connect(host=dbparams['ip'],port=int(dbparams['port']),user=dbuser,passwd=dbpass,db="information_schema")
                except:
                    print(sys.exc_info())
                    print("Connection to server "+dbparams['name']+" failed!")
                else:
                    hosts_dict[dbparams['name']] = dbparams
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
                        #for data in cur_slave_hosts:
                        #    print(data)

                    # Check if "show slave status" returned at least 1 row
                    if cur_is_slave.rowcount > 0:
                        is_slave = 1

                    # Infer node role based on the queries above
                    if has_slaves == 1 and is_slave == 0:
                        print("Host "+dbparams['name']+" seems to be master")
                    elif has_slaves == 1 and is_slave == 1:
                        print("Host "+dbparams['name']+" seems to be an intermediary node")
                    elif has_slaves == 0 and is_slave == 1:
                        print("Host "+dbparams['name']+" seems to be a slave")
                    db.close()
    except:
        print("Exception opening hostlist files")
    else:
        return hosts_dict



def migrate_slaves(current_master,candidate_master,slaves_list,hosts_dict):

    def save_slaves_metadata(name,cur_is_slave):
        sls_out = cur_is_slave.fetchone()
        slave_relay_masterbinlog[name] = sls_out[9]

    slave_relay_masterbinlog={}
    idx = 0
    dbconar={}
    hostlist = [current_master] + [candidate_master] + slaves_list

    # Validate that there are no inconsistencies in the list and roles of then servers provided
    if current_master == candidate_master or current_master in slaves_list or candidate_master in slaves_list:
        print("Current master and candidate master cannot be the same. Also none of the masters could be in the slave list")
        return None

    # Check that each of the hosts specified is already registered
    for server in hostlist:
        if server not in hosts_dict.keys():
            print(server+" is not a registered server")
            return None
        else:
            dbconar[server] = MySQLdb.connect(host=hosts_dict[server]['ip'],port=int(hosts_dict[server]['port']),user=dbuser,passwd=dbpass,db="information_schema")

    # Retrieve relay master binlog for the candidate master and the slaves
    for slave in slaves_list+[candidate_master]: 
        has_slaves = 0
        is_slave = 0
        cur_is_slave = dbconar[slave].cursor()
        cur_is_slave.execute("show slave status")
        save_slaves_metadata(hosts_dict[slave]['name'],cur_is_slave)

    print(slave_relay_masterbinlog)

    # Retrieve master's active binlog
    cursor = dbconar[current_master].cursor()
    cursor.execute("show master status")
    master_cur_coord = cursor.fetchone()
    master_cur_coord_file = master_cur_coord[0]
    print(master_cur_coord_file)

    # Check that all slaves are on the same master binlog
    for masterbinlog in slave_relay_masterbinlog:
        if idx == 0:
           prevbl = slave_relay_masterbinlog[masterbinlog]
        if prevbl != slave_relay_masterbinlog[masterbinlog]:
            print("Slaves are not all applying the same master binlog")
            break
        else:
            prevbl = slave_relay_masterbinlog[masterbinlog]
        idx+=1
    if idx == slave_relay_masterbinlog.__len__():
        print("All slaves are at the same binlog")

    # Check that the master's active binlog matches the slave relay binlog
    if prevbl == master_cur_coord_file:
        print("The slaves are reading the master active binlog file")
        print("Stopping all slaves sql_thread")

    else:
        print("The slaves are NOT reading the master active binlog file")

    # Stop all slaves and start them UNTIL master_binlog+1
    for slave in slaves_list+[candidate_master]: 
        cur_is_slave = dbconar[slave].cursor()
        cur_is_slave.execute("stop slave sql_thread")
        cur_is_slave.execute("stop slave sql_thread")

    for conn in dbconar:
        dbconar[conn].close()

        
def main():
    hosts_dict = load_slaves()
    migrate_slaves('master','slave3',['slave1','slave2'],hosts_dict)

if __name__ == '__main__':
    main()


#!/usr/bin/python

import MySQLdb
import csv
import sys
import time
import os

hostlist='hostlist.csv'
dbuser = "percona"
dbpass = "percona"
repluser="msandbox"
replpass="msandbox"

# Main definition - constants
menu_actions  = {}

## <<< Flush binlog at the master at the beginning to have 1Gb of time to perform the changes

def load_slaves():

    hosts_dict={}
    dbconar = {}
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
                    dbconar[dbparams['name']] = db
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
    except:
        print("Exception opening hostlist files")
    else:
        return hosts_dict,dbconar



def ret_relay_master_binlog(slaves_list,dbconar):

    if len(slaves_list) == 0:
        slaves = raw_input(">> slave list: ")
        slaves_list = []
        for slave in slaves.split(','):
            slaves_list.append(slave)
        
    # Retrieve relay master binlog for the candidate master and the slaves
    slave_relay_masterbinlog = {}
    for slave in slaves_list: 
        cur_is_slave = dbconar[slave].cursor()
        cur_is_slave.execute("show slave status")
        if cur_is_slave.rowcount > 0:
            sls_out = cur_is_slave.fetchone()
            slave_relay_masterbinlog[slave] = {"relay_master_log_file":sls_out[9],"slave_io_running":sls_out[10],"slave_sql_running":sls_out[11],"exec_master_log_pos":sls_out[21],"master_host":sls_out[1],"master_port":sls_out[3]}
            print("Server "+slave+"-> master_host: "+slave_relay_masterbinlog[slave]['master_host']+":"+slave_relay_masterbinlog[slave]['master_port'].__str__()+" - relay_master_log_file: "+slave_relay_masterbinlog[slave]['relay_master_log_file']+" - exec_master_log_pos: "+slave_relay_masterbinlog[slave]['exec_master_log_pos'].__str__())
        else:
            print("Server "+slave+" is not a slave")
            
            

    return slave_relay_masterbinlog
    

def migrate_back(hosts_dict,dbconar):
    
    current_master = raw_input(">> current master: ")
    candidate_master = raw_input(">> candidate master: ")
    slaves = raw_input(">> slave list: ")

    slaves_list = []
    for slave in slaves.split(','):
        slaves_list.append(slave)

    for slave in slaves_list:
        pass ## << ToDo

def migrate_slaves(hosts_dict,dbconar):
    
    current_master = raw_input(">> current master: ")
    candidate_master = raw_input(">> candidate master: ")
    slaves = raw_input(">> slave list: ")

    slaves_list = []
    for slave in slaves.split(','):
        slaves_list.append(slave)

    def compare_slaves_binlog():
        idx = 0
        # Check that all slaves are on the same master binlog
        for masterbinlog in slave_relay_masterbinlog:
            if idx == 0:
               prevbl = slave_relay_masterbinlog[masterbinlog]['relay_master_log_file']
            if prevbl != slave_relay_masterbinlog[masterbinlog]['relay_master_log_file']:
                print("Slaves are not all applying the same master binlog")
                break
            else:
                prevbl = slave_relay_masterbinlog[masterbinlog]['relay_master_log_file']
            idx+=1
        if idx == slave_relay_masterbinlog.__len__():
            print("All slaves are at the same binlog")
            response = prevbl
        else:
            response = None
        return response

    def ret_master_status(server):
        # Retrieve master's active binlog
        cursor = dbconar[server].cursor()
        cursor.execute("show master status")
        master_cur_coord = cursor.fetchone()
        master_cur_coord_file = master_cur_coord[0]
        return master_cur_coord[0],master_cur_coord[1]


    ## Maint
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
    #    else:
            #dbconar[server] = MySQLdb.connect(host=hosts_dict[server]['ip'],port=int(hosts_dict[server]['port']),user=dbuser,passwd=dbpass,db="information_schema")

    # Get slaves relay binlog
    slave_relay_masterbinlog = ret_relay_master_binlog(slaves_list+[candidate_master],dbconar)
    # Validate if all slaves are applying the same binlog
    slavesbinlog = compare_slaves_binlog()
    if slavesbinlog == None:
        return 1
    print("")
    # Get master binlog coordinates
    master_cur_coord_file,master_cur_coord_file_pos = ret_master_status(current_master)
    print("Server "+current_master+"-> Current binlog: "+master_cur_coord_file+" - binlog position: "+master_cur_coord_file_pos.__str__())
    print("")
    #print(master_cur_coord_file)
    #print(master_cur_coord_file_pos)


    # Check that the master's active binlog matches the slave relay binlog
    if slavesbinlog !=None and slavesbinlog == master_cur_coord_file:
        print("The slaves are reading the master active binlog file")
        print("Stopping all slaves sql_thread")
        # Stop all slaves and start them UNTIL master_binlog+1
        for slave in slaves_list+[candidate_master]: 
            cur_is_slave = dbconar[slave].cursor()
            cur_is_slave.execute("stop slave sql_thread")


        # Get slaves relay binlog again
        slave_relay_masterbinlog = ret_relay_master_binlog(slaves_list+[candidate_master],dbconar)
        # Validate if all slaves were stopped at the same binlog
        slavesbinlog = compare_slaves_binlog()
        print("")

        # if they are all at the same master binlog, stop the slave io thread
        if slavesbinlog != None:
            # Retrieve master current binlog
            master_cur_coord_file,master_cur_coord_file_pos = ret_master_status(current_master)

            # Extract binlog number portion
            master_binlog_num = int(master_cur_coord_file.split('.')[1])
            #print(master_binlog_num)
            target_mas_binlog_num = master_binlog_num + 1
            # Prepend binlog prefix
            target_mas_binlog = 'mysql-bin.'+('0' * (6-len(target_mas_binlog_num.__str__())))+target_mas_binlog_num.__str__()
            print("Slaves will stop at binlog "+target_mas_binlog)
            
            # Run START SLAVE .. UNTIL
            for slave in slaves_list+[candidate_master]: 
                cur_is_slave = dbconar[slave].cursor()
                cur_is_slave.execute("start slave until MASTER_LOG_FILE='"+target_mas_binlog+"',MASTER_LOG_POS=107")

            # Force a log switch on the master
            print("Forcing binlog switch on the master..")
            force_master_sitch = dbconar[current_master].cursor()
            force_master_sitch.execute("flush logs")

            # Wait until all slave SQL threads were stopped
            sql_threads_stop = 1
            while sql_threads_stop > 0:
                print("Waiting for all SQL slave threads to stop")
                sql_threads_stop = 0
                slave_relay_masterbinlog = ret_relay_master_binlog(slaves_list+[candidate_master],dbconar)
                for slave in slave_relay_masterbinlog:
                    if slave_relay_masterbinlog[slave]['slave_sql_running'] == 'Yes':
                        sql_threads_stop += 1
                time.sleep(2)

            # Make "candidate_master" a master or [slaves_list]
            error=0
            binlog_file,binlog_pos = ret_master_status(candidate_master)
            for slave in slaves_list:
                change_master_stmt = "change master to master_host='"+hosts_dict[candidate_master]['ip']+"',master_port="+hosts_dict[candidate_master]['port']+",master_log_file='"+binlog_file+"',master_log_pos="+binlog_pos.__str__()+",master_user='"+repluser+"',master_password='"+replpass+"'"
                print(change_master_stmt)
                try:
                    cur_change_master = dbconar[slave].cursor()
                    cur_change_master.execute("stop slave")
                    cur_change_master.execute(change_master_stmt)
                except:
                    print("Slaves repoint failed - SQL threads for all servers will remain stopped!")
                    error = 1
            if error == 0:
                print("Starting slaves")
                for slave in slaves_list+[candidate_master]:
                    cur_start_slave = dbconar[slave].cursor()
                    cur_start_slave.execute("start slave")
                




        else:
            print("Slaves were not stopped on the same binlog")

    else:
        print("Slaves are not all reading the master active binlog file OR The slaves are NOT reading the master active binlog file")
    # Close all database connections
        return 1

# Main menu
def main_menu(hosts_dict,dbconar):
    
    print "Migrator"
    print "************\n"
    print "1. Show slave status"
    print "2. Migrate slaves to intermediary"
    print "3. Migrate slaves back to main master"
    print "0. Quit"
    choice = raw_input(" >>  ")
    if choice == "1":
        ret_relay_master_binlog([],dbconar)
    elif choice == "2":
        migrate_slaves(hosts_dict,dbconar)
    elif choice == "3":
        migrate_back(hosts_dict,dbconar)
    elif choice == "0":
        exit
    else:
        print("Invalid option")
    return

def close_conns(dbconar):
    for conn in dbconar:
        dbconar[conn].close()
        
def main():
    hosts_dict,dbconar = load_slaves()
    print("")
    print("")
    #migrate_slaves('master','slave3',['slave1','slave2'],hosts_dict)
    main_menu(hosts_dict,dbconar)
    print("Closing database connections...")
    close_conns(dbconar)


if __name__ == '__main__':
    main()


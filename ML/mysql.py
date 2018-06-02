import pymysql
import datetime

def createDbConn():
    # Open database connection
    db = pymysql.connect("localhost","thesis","Clustering453!","thesisdb" )
    return db


def executeTrx(db,op,valuetuple):
    lii = None
    try:
        cursor = db.cursor()
        cursor.execute(op,valuetuple)
        db.commit()
    except:
        print(op,' failed')
        db.rollback()
    else:
        lii = cursor.lastrowid
    return lii


def getVersion(db):
    # prepare a cursor object using cursor() method
    cursor = db.cursor()

    # execute SQL query using execute() method.
    cursor.execute("SELECT VERSION()")

    # Fetch a single row using fetchone() method.
    data = cursor.fetchone()
    return "Database version: "+str(data[0])


def insertRun(db):
    op = "insert into run(start_date) values (%s)"
    runid = executeTrx(db,op,(datetime.datetime.now()))
    return runid

def insertDataset(db,runid,features,total_samples,linear_samples_perc,repeated_samples_perc,group_number,outliers_perc,uniform_features,standard_features):
    op = "insert into dataset(run_id,features,total_samples,linear_samples_perc,repeated_samples_perc,group_number,outliers_perc,uniform_features,standard_features) values (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    datasetid = executeTrx(db,op,(runid,features,total_samples,linear_samples_perc,repeated_samples_perc,group_number,outliers_perc,uniform_features,standard_features))
    return datasetid

def insertDatasetValidation(db,runid,features,total_samples,linear_samples_perc,repeated_samples_perc,group_number,outliers_perc,outliersbyperp_perc):
    op = "insert into dataset(run_id,features,total_samples,linear_samples_perc,repeated_samples_perc,group_number,outliers_perc,uniform_features,standard_features) values (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    datasetvalidationid = executeTrx(db,op,(runid,features,total_samples,linear_samples_perc,repeated_samples_perc,group_number,outliers_perc,outliersbyperp_perc))
    return datasetid

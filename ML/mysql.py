import pymysql
import datetime
import sys

def createDbConn():
    # Open database connection
    db = pymysql.connect("localhost","thesis","Clustering453!","thesisdb" )
    return db


def executeTrx(db,op,valuetuple):
    # Enable or disable writes to the DB
    write=1

    lii = None
    if write == 1:
        try:
            cursor = db.cursor()
            cursor.execute(op,valuetuple)
            db.commit()
        except:
            print(op,' failed')
            print("Unexpected error:", sys.exc_info()[0])
            db.rollback()
            raise
        else:
            lii = cursor.lastrowid
    else:
        print('############### WRITE DISABLED !! ###################')
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

def insertDatasetValidation(db,runid,datasetid,features,total_samples,linear_samples_perc,repeated_samples_perc,group_number,outliers_perc,outliersbyperp_perc):
    op = "insert into dataset_validation(run_id,dataset_id,features,total_samples,linear_samples_perc,repeated_samples_perc,group_number,outliers_perc,outliersbyperp_perc) values (%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    datasetvalidationid = executeTrx(db,op,(runid,datasetid,features,total_samples,linear_samples_perc,repeated_samples_perc,group_number,outliers_perc,outliersbyperp_perc))
    return datasetvalidationid


def insertClusMetrics(db,datasetid,algorithm,total_clusters,single_element_clusters,samples_not_considered,elap_time,silhouette_score,calinski_harabaz_score,wb_index,dunn_index,davies_bouldin_score):

    # Truncate calinski_harabaz_score value if it exceeds the datatype precision
    if calinski_harabaz_score != None and calinski_harabaz_score > 999999999999999.9999:
        calinski_harabaz_score = '999999999999999.9'
    op = "insert into clustering_metric(dataset_id,algorithm,total_clusters,single_element_clusters,samples_not_considered,elap_time,silhouette_score,calinski_harabaz_score,wb_index,dunn_index,davies_bouldin_score) values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    clusmetricsid = executeTrx(db,op,(datasetid,algorithm,total_clusters,single_element_clusters,samples_not_considered,elap_time,silhouette_score,calinski_harabaz_score,wb_index,dunn_index,davies_bouldin_score))
    return clusmetricsid

def updateDatasetClusAlg(db,datasetid,winner):
    op = "update dataset set winner_clus_alg=%s where id=%s"
    datasetid = executeTrx(db,op,(winner,datasetid))
    return datasetid

def insertRIMetrics(db,datasetid,clustering_metric_id,algorithm,total_rules,elap_time,auc):
    op = "insert into rule_ind_metric(dataset_id,clustering_metric_id,algorithm,total_rules,elap_time,auc) values (%s,%s,%s,%s,%s,%s)"
    rimetricsid = executeTrx(db,op,(datasetid,clustering_metric_id,algorithm,total_rules,elap_time,auc))
    return rimetricsid

def updateDatasetRIAlg(db,datasetid,winner):
    op = "update dataset set winner_ri_alg=%s where id=%s"
    datasetid = executeTrx(db,op,(winner,datasetid))
    return datasetid

def updateRun(db,runid):
    op = "update run set end_time=%s where id=%s"
    datasetid = executeTrx(db,op,(datetime.datetime.now(),runid))
    return runid

def insertDatasetClusFinalistsR1(db,datasetid,winalg,metricsmask,iswinner):
    op = "insert into dataset_clus_finalists(dataset_id,algorithm,silhouette,calinski_harabaz,dunn,wb,davies_bouldin,winner) values (%s,%s,%s,%s,%s,%s,%s,%s)"
    dataseclusfinalistid = executeTrx(db,op,(datasetid,winalg,metricsmask['silhouette_score'],metricsmask['calinski_harabaz'],metricsmask['dunn'],metricsmask['wb'],metricsmask['davies_bouldin'],iswinner))
    return dataseclusfinalistid

def updateDatasetClusFinalistsR2(db,datasetid,winalg,metricsmask,iswinner):
    op = "update dataset_clus_finalists set time=%s, sin_ele_clus=%s, ignored_samples=%s,winner=%s where dataset_id=%s and algorithm=%s"
    dataseclusfinalistid = executeTrx(db,op,(metricsmask['time'],metricsmask['sin_ele_clus'],metricsmask['ignored_samples'],iswinner,datasetid,winalg))
    return dataseclusfinalistid


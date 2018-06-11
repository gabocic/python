-- MySQL dump 10.13  Distrib 5.7.21-21, for debian-linux-gnu (x86_64)
--
-- Host: localhost    Database: thesisdb
-- ------------------------------------------------------
-- Server version	5.7.21-21

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
/*!50717 SELECT COUNT(*) INTO @rocksdb_has_p_s_session_variables FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'performance_schema' AND TABLE_NAME = 'session_variables' */;
/*!50717 SET @rocksdb_get_is_supported = IF (@rocksdb_has_p_s_session_variables, 'SELECT COUNT(*) INTO @rocksdb_is_supported FROM performance_schema.session_variables WHERE VARIABLE_NAME=\'rocksdb_bulk_load\'', 'SELECT 0') */;
/*!50717 PREPARE s FROM @rocksdb_get_is_supported */;
/*!50717 EXECUTE s */;
/*!50717 DEALLOCATE PREPARE s */;
/*!50717 SET @rocksdb_enable_bulk_load = IF (@rocksdb_is_supported, 'SET SESSION rocksdb_bulk_load = 1', 'SET @rocksdb_dummy_bulk_load = 0') */;
/*!50717 PREPARE s FROM @rocksdb_enable_bulk_load */;
/*!50717 EXECUTE s */;
/*!50717 DEALLOCATE PREPARE s */;

--
-- Current Database: `thesisdb`
--

CREATE DATABASE /*!32312 IF NOT EXISTS*/ `thesisdb` /*!40100 DEFAULT CHARACTER SET latin1 */;

USE `thesisdb`;

--
-- Table structure for table `clustering_metric`
--

DROP TABLE IF EXISTS `clustering_metric`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `clustering_metric` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `dataset_id` bigint(20) DEFAULT NULL,
  `algorithm` varchar(15) DEFAULT NULL,
  `total_clusters` int(11) DEFAULT NULL,
  `single_element_clusters` int(11) DEFAULT NULL,
  `samples_not_considered` bigint(20) DEFAULT NULL,
  `elap_time` decimal(19,4) DEFAULT NULL,
  `silhouette_score` decimal(19,4) DEFAULT NULL,
  `calinski_harabaz_score` decimal(19,4) DEFAULT NULL,
  `wb_index` decimal(19,4) DEFAULT NULL,
  `dunn_index` decimal(19,4) DEFAULT NULL,
  `davies_bouldin_score` decimal(19,4) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=2542 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `dataset`
--

DROP TABLE IF EXISTS `dataset`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `dataset` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `run_id` bigint(20) DEFAULT NULL,
  `total_samples` bigint(20) DEFAULT NULL,
  `features` int(11) DEFAULT NULL,
  `linear_samples_perc` decimal(4,1) DEFAULT NULL,
  `repeated_samples_perc` decimal(4,1) DEFAULT NULL,
  `group_number` smallint(6) DEFAULT NULL,
  `outliers_perc` decimal(4,1) DEFAULT NULL,
  `uniform_features` int(11) DEFAULT NULL,
  `standard_features` int(11) DEFAULT NULL,
  `winner_clus_alg` varchar(15) DEFAULT NULL,
  `winner_ri_alg` varchar(15) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=466 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `dataset_validation`
--

DROP TABLE IF EXISTS `dataset_validation`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `dataset_validation` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `run_id` bigint(20) DEFAULT NULL,
  `total_samples` bigint(20) DEFAULT NULL,
  `features` int(11) DEFAULT NULL,
  `linear_samples_perc` decimal(4,1) DEFAULT NULL,
  `repeated_samples_perc` decimal(4,1) DEFAULT NULL,
  `group_number` smallint(6) DEFAULT NULL,
  `outliers_perc` decimal(4,1) DEFAULT NULL,
  `outliersbyperp_perc` decimal(4,1) DEFAULT NULL,
  `dataset_id` bigint(20) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=466 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `old_clustering_metric`
--

DROP TABLE IF EXISTS `old_clustering_metric`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `old_clustering_metric` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `dataset_id` bigint(20) DEFAULT NULL,
  `algorithm` varchar(15) DEFAULT NULL,
  `total_clusters` int(11) DEFAULT NULL,
  `single_element_clusters` int(11) DEFAULT NULL,
  `samples_not_considered` bigint(20) DEFAULT NULL,
  `elap_time` decimal(11,4) DEFAULT NULL,
  `silhouette_score` decimal(11,4) DEFAULT NULL,
  `calinski_harabaz_score` decimal(11,4) DEFAULT NULL,
  `wb_index` decimal(11,4) DEFAULT NULL,
  `dunn_index` decimal(11,4) DEFAULT NULL,
  `davies_bouldin_score` decimal(11,4) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=505 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `rule_ind_metric`
--

DROP TABLE IF EXISTS `rule_ind_metric`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `rule_ind_metric` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `dataset_id` bigint(20) DEFAULT NULL,
  `clustering_metric_id` bigint(20) DEFAULT NULL,
  `algorithm` char(6) DEFAULT NULL,
  `total_rules` int(11) DEFAULT NULL,
  `elap_time` decimal(11,4) DEFAULT NULL,
  `auc` decimal(11,4) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=805 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `run`
--

DROP TABLE IF EXISTS `run`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `run` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `start_date` datetime DEFAULT NULL,
  `end_time` datetime DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=78 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!50112 SET @disable_bulk_load = IF (@is_rocksdb_supported, 'SET SESSION rocksdb_bulk_load = @old_rocksdb_bulk_load', 'SET @dummy_rocksdb_bulk_load = 0') */;
/*!50112 PREPARE s FROM @disable_bulk_load */;
/*!50112 EXECUTE s */;
/*!50112 DEALLOCATE PREPARE s */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2018-06-10 22:23:07

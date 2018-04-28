-- MySQL dump 10.13  Distrib 5.7.21, for Linux (x86_64)
--
-- Host: localhost    Database: thesisdb
-- ------------------------------------------------------
-- Server version	5.7.21-0ubuntu0.16.04.1

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

--
-- Table structure for table `clustering_metrics`
--

DROP TABLE IF EXISTS `clustering_metrics`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `clustering_metrics` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `dataset_id` bigint(20) DEFAULT NULL,
  `algorithm` char(6) DEFAULT NULL,
  `total_clusters` int(11) DEFAULT NULL,
  `single_element_clusters` int(11) DEFAULT NULL,
  `samples_not_considered` bigint(20) DEFAULT NULL,
  `elap_time` decimal(11,4) DEFAULT NULL,
  `silhouette_score` decimal(11,4) DEFAULT NULL,
  `calinski_harabaz_score` decimal(11,4) DEFAULT NULL,
  `wb_index` decimal(11,4) DEFAULT NULL,
  `dunn_index` decimal(11,4) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
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
  `random_samples_perc` decimal(4,1) DEFAULT NULL,
  `linear_samples_perc` decimal(4,1) DEFAULT NULL,
  `repeated_samples_perc` decimal(4,1) DEFAULT NULL,
  `group_number` smallint(6) DEFAULT NULL,
  `outliers_perc` decimal(4,1) DEFAULT NULL,
  `uniform_features` int(11) DEFAULT NULL,
  `standard_features` int(11) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
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
  `random_samples_perc` decimal(4,1) DEFAULT NULL,
  `linear_samples_perc` decimal(4,1) DEFAULT NULL,
  `repeated_samples_perc` decimal(4,1) DEFAULT NULL,
  `group_number` smallint(6) DEFAULT NULL,
  `outliers_perc` decimal(4,1) DEFAULT NULL,
  `outliersbyperp_perc` decimal(4,1) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
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
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
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
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2018-04-28 10:53:24

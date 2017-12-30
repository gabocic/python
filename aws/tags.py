import sys
import boto3

class persistclass:

	def setnodestatus(self,hostname,status):
		ec2 = boto3.resource('ec2')
		try:
			ec2.create_tags(DryRun=False,Resources=[hostname],Tags=[{'Key':'adStatus','Value':status.__str__()}])
		except:
			print "setnodestatus - Call failed"
			print sys.exc_info()
		else:
			pass

	def tagInstance(self,hostname,role,newproject,idx):
		ec2 = boto3.resource('ec2')
		ec2.create_tags(DryRun=False,Resources=[hostname],Tags=[{'Key':'clustername','Value':newproject},{'Key':'rolename','Value':role},{'Key':'adStatus','Value':'0'},{'Key':'Name','Value':newproject+'-'+role+'-'+idx.__str__()}])
		#return query


	def getnodesstatus(self,dbcluster):
		running = 0
		notready = 0
		ec2 = boto3.resource('ec2')
		try:
			result = ec2.instances.filter(Filters=[{'Name':'tag:clustername','Values':[dbcluster]}])
		except:
			print "getnodesstatus - Query failed!"
		else:
			for instance in result:
				for tag in instance.tags:
					if tag['Key'] == 'adStatus':
						adstatus = tag['Value']

				if instance.state['Name'] == "running":
					if adstatus == '0':
						running = running + 1
				else:
					notready = notready + 1
			nodesstatus = [notready,running]
			return nodesstatus

	def getnodes(self,dbcluster):
		allnodes=[]
		ec2 = boto3.resource('ec2')
		try:
			result = ec2.instances.filter(Filters=[{'Name':'tag:clustername','Values':[dbcluster]}])
		except:
			print "getnodes - Query failed!"
		else:
			for instance in result:
				for tag in instance.tags:
					if tag['Key'] == 'rolename':
						role = tag['Value']
					if tag['Key'] == 'Name':
						hostname = tag['Value']
				allnodes.append({'ipadd':instance.private_ip_address,'role':role,'hostname':hostname})
			return allnodes

	def getsoftreadynodes(self,dbcluster):
		ec2 = boto3.resource('ec2')
		try:
			result = ec2.instances.filter(Filters=[{'Name':'instance-state-name','Values':['running']},{'Name':'tag:clustername','Values':[dbcluster]},{'Name':'tag:adStatus','Values':['0']}])
		except:
			print "getsoftreadynodes - Query failed!"
		else:
			return result

	def getprimaries(self,dbcluster):
		allnodes=[]
		ec2 = boto3.resource('ec2')
		try:
			result = ec2.instances.filter(Filters=[{'Name':'tag:clustername','Values':[dbcluster]},{'Name':'tag:rolename','Values':['RS*']}])
		except:
			print "getnodes - Query failed!"
		else:
			for instance in result:
				for tag in instance.tags:
					if tag['Key'] == 'rolename':
						role = tag['Value']
					if tag['Key'] == 'Name':
						hostname = tag['Value']
				allnodes.append({'role':role,'hostname':hostname})
		primaries={}
		for rsmembers in allnodes:
			if rsmembers["role"] not in primaries.keys():
				primaries[rsmembers["role"]] = rsmembers["hostname"]
		return primaries
	
	
	def getcfgservers(self,dbcluster):
		allnodes=[]
		ec2 = boto3.resource('ec2')
		try:
			result = ec2.instances.filter(Filters=[{'Name':'tag:clustername','Values':[dbcluster]},{'Name':'tag:rolename','Values':['CFG']}])
		except:
			print "getcfgservers - Query failed!"
		else:
			for instance in result:
				for tag in instance.tags:
					if tag['Key'] == 'Name':
						hostname = tag['Value']
				allnodes.append(hostname)
		return allnodes

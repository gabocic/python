import sys
from time import sleep
import string
import boto3
import sys

amid="ami-xxxxxxxx"
giversionid=3
quantity=1

class instancegen:

    def create_instance(self,flavor):
	def request_instance(flavor):
	    retry=1
	    ec2 = boto3.resource('ec2')
	    v_mincount=1
	    v_maxcount=1
	    v_insttype=flavor
	    v_secgroup=['sg-xxxxxxxx']
	    v_keyname='mykey'
	    #v_vpc
	    request_counter = 0
	    while request_counter < retry:
	        try:
	    	    result = ec2.create_instances(DryRun=False,ImageId=amid,MinCount=v_mincount,MaxCount=v_maxcount,InstanceType=v_insttype,SecurityGroupIds=v_secgroup,KeyName=v_keyname,SubnetId='subnet-efb2859b')
		except:
		    print "Error creating EC2 instance:",sys.exc_info()[0]
		else:
		    print result
		    break
		request_counter = request_counter + 1 
		sleep(3)
	    if request_counter == retry:
			print "Failed to create a vm instance after "+retry.__str__()+" retries..."
	                jres = ["failure"]
	    else:
			print "instance successfully created"
	                jres = result
            return jres


	### Main ###
	hostdata = request_instance(flavor)
	errcode = 1
	if hostdata[0] == "failure":
        	return {"errorcode":errcode,"instname":"failure","instip":"failure"}
	else:
        	return {"errorcode":errcode,"instname":hostdata[0].id,"instip":hostdata[0].private_ip_address}

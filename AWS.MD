# Connect to EC2
You need the .pem file, in this case the example is ~/dvl/beTheChange/scrapping.pem then you use:

```ssh -i ~/dvl/beTheChange/scrapping.pem ubuntu@ec2-52-27-52-177.us-west-2.compute.amazonaws.com```

To know the machine to use and the username you can go to the EC2 Console and find the instance you are interested in,
then right click and use the 'Connect' option which will tell you what is the exact name to use.

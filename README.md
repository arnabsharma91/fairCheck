# fairCheck
This repository contains the tool for testing a given ML model for a specific fairness definition. This definition can be specified by the user by using ```assume-assert``` format. This work has been published in the ICTSS 2020 in the paper named '[Automatic Fairness Testing of Machine Learning Models](https://link.springer.com/chapter/10.1007/978-3-030-64881-7_16)'. We will provide a link of our [paper]() as soon as the proceedings are available.

## Contributors
[Arnab Sharma](https://en.cs.uni-paderborn.de/sms/team/people/arnab-sharma), [Heike Wehrheim](https://en.cs.uni-paderborn.de/sms/team/people/heike-wehrheim)

## Requirements
We have used Python3.6 to develop our tool. Hence, to run our tool you need to install Python3. Also, you need to install scipy, scikit-learn, pandas, numpy, parsimonious and SMT solver z3. These can be installed using the ```pip``` command. For example:
```
pip install <package-name>
```

## Usage
Our tool needs to have the input schema which describes the input data instance for the given ML model under test. This schema is provided as an XML file. The XML file can be written by the tester. We have provided a sample XML file (```dataInput.xml```) to give you an idea how to write such schema XML file. However, writing such an XML file by hand can be very tedious, hence, we have provided a converter which takes as input the dataset as .csv file and automatically creates such an XML file in our desired format. For example, if you want to create the schema XML file of the Adult dataset which has been saved inside the [Datasets](https://github.com/arnabsharma91/fairCheck/tree/master/Datasets) folder, you can run the following:
```
python Dataframe2XML.py Datasets/Adult.csv
```
After creating such a schema file, next step is to run our tool for testing fairness. We have created a sample file ```testCase.py``` which describes how to test a given ML model using our testing approach. In the begining, you have to fix some parameters for our tools 

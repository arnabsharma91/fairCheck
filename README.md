# fairCheck
This repository contains the tool for testing a given ML model for a specific fairness definition. This definition can be specified by the user by using ```assume-assert``` format. This work has been published in the ICTSS 2020 in the paper named 'Automatic Fairness Testing of Machine Learning Models'. We will provide a link of our [paper]() as soon as the proceedings are available.

## Contributors
[Arnab Sharma](https://en.cs.uni-paderborn.de/sms/team/people/arnab-sharma), [Heike Wehrheim](https://en.cs.uni-paderborn.de/sms/team/people/heike-wehrheim)

## Requirements
We have used Python3.6 to develop our tool. Hence, to run our tool you need to install Python3. Also, you need to install scipy, scikit-learn, pandas, numpy, parsimonious and SMT solver z3. These can be installed using the ```pip``` command. For example: <br>
```
pip install <package-name>
```

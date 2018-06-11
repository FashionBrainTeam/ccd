# CCD: Classification of fashion time series

EU project 732328: "Fashion Brain".

D5.4: " The classification algorithm and its evaluation on fashion time series".

## Getting started

### Installation:
``` bash 
git clone https://github.com/eXascaleInfolab/ccd.git}
cd ccd/
```

### Description of the "CCD" package
The ”CCD” package contains the following files:
- cd_cluster.py: the implementation of CCD algorithm
- input folder: This folder contains the running example input matrix in the file ”example.txt”
- result folder:This folder contains the results of the CCD on the running example 

 

### Running the code 
In order to run an experiment, the cd_cluster.py file is used. The arguments needed
for the cd_cluster.py are the following:
- The path for the input file
- The path for the output file
- Number of rows n, which takes any integer number
- Number of columns m, which takes any integer number
- Number of truncated columns k, which takes any integer number and needs to be less than m


Example: To run an experiment of CCD with the following parameters:
- input file=./Input/example.txt
- output file= ./Result/classes.txt
- number of rows= 8
- number of cols= 4
- number of truncated cols = 2

The corresponding command line would be the following:
``` bash 
python cd_cluster.py ./Input/example.txt ./Result/classes.txt 8 4 2
```

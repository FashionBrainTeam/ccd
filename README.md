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
- cd cluster.py: the implementation of CCD algorithm
- Input folder: This folder contains the running example input matrix in the file ”example.txt”
- Result folder:This folder contains the results of the CCD on the running example 

 

### Running the code 
In order to run an experiment, the cd cluster.py file is used. The arguments needed
for the cd cluster.py file are the following:
- The path for the input file
- The path for the output file
- number of rows n, which takes any integer number
- number of columns m, which takes any integer number
- number of truncated columns k, which takes any integer number and needs to be less than m


Example: To run an experiment of CCD with the following parameters:
- input file=./Input/example.txt
- output file= ./Result/classes.txt
- number of rows= 8
- number of cols=4
- number of truncated cols = 2
the corresponding command line would be the following:
    ``` bash 
       python cd_cluster.py ./Input/example.txt ./Result/classes.txt 8 4 2
    ```

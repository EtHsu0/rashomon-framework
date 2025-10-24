# DPF: Dynamic Programming Approach to Fair and Optimal Decision Trees
By: Jacobus G.M. van der Linden

The code in this repository implements the DPF algorithm for constructing fair and optimal binary classification trees. It minimizes misclassification score, while respecting a demographic parity fairness constraint, for a given size (depth, number of nodes). It can also generate the full Pareto front while optimizing both fairness and accuracy. 

Based on the MurTree algorithm developed by Emir Demirović et al. ([source](https://bitbucket.org/EmirD/murtree/src/master/)) and its biobjective variant ([source](https://bitbucket.org/EmirD/murtree-bi-objective/src/master/)).

Details about the algorithm can be found in our paper. Please cite our paper if you use our code:

Jacobus G.M. van der Linden, Mathijs M. de Weerdt, and Emir Demirović. "Fair and Optimal Decision Trees, a Dynamic
 Programming Approach." In _Advances in NeurIPS-22_, 2022.

## Compiling
The code can be compiled on Windows or Linux by using cmake. For Windows users, cmake support can be installed as an extension of Visual Studio and then this repository can be imported as a CMake project.

For Linux users, they can use the following commands:

```sh
mkdir build
cd build
cmake ..
cmake --build .
```
This has been tested with gcc 9.4. Older versions may not support the C++17 standard

## Running
After DPF is built, the following command can be used (for example) to find a fair and optimal tree for the Student-Portuguese dataset:
```sh
./DPF -file ../data/student-por-binarized.csv -stat-test-value 0.01 -max-depth 3 -max-num-nodes 7
```

Run the program without any parameters to see a full list of the available parameters.

For benchmarking another program is also built: DPFBenchmark

## Docker
Alternatively, docker can be used to build and run DPF:
```
docker build -t dpf .
docker container run -it dpf /dpf/build/DPF -file /dpf/data/student-por-binarized.csv -stat-test-value 0.01 -max-depth 3 -max-num-nodes 7
```

## Data
The datasets are included in the data folder (except those for which redistribution is not allowed). Source and license information is described per dataset in the data/sources.txt file.




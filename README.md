# tidb-vector-bench

## Getting Started

Install requirements:

```shell
# cd tidb-vector-bench
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```

Download data set:

```shell
# cd tidb-vector-bench
wget https://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5
```

Run (load data set):

```shell
python3 main.py local load
```

Run (perform a vector search using EXPLAIN ANALYZE):

```shell
python3 main.py local test
```

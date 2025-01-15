import os
import dotenv
import sys
import h5py
import numpy
import peewee
import pymysql
import tabulate
import subprocess
from tidb_vector.peewee import VectorField, VectorAdaptor
from tidb_vector.utils import encode_vector
import tidb_vector


def exit_print_usage():
    print(
        "Usage: python main.py <local|dev|staging|production> <load|test|clean|connect>"
    )
    sys.exit(1)


if len(sys.argv) < 3:
    exit_print_usage()

env = sys.argv[1]
action = sys.argv[2]

dotenv.load_dotenv()


if env == "dev":
    mysql_db = peewee.MySQLDatabase(
        "test",
        host=os.getenv("DEV_HOST"),
        port=4000,
        user=os.getenv("DEV_USER"),
        passwd=os.getenv("DEV_PASS"),
        ssl_verify_cert=True,
        ssl_verify_identity=True,
        ssl_ca="/etc/ssl/cert.pem",
    )
elif env == "staging":
    mysql_db = peewee.MySQLDatabase(
        "test",
        host=os.getenv("STAGING_HOST"),
        port=4000,
        user=os.getenv("STAGING_USER"),
        passwd=os.getenv("STAGING_PASS"),
        ssl_verify_cert=True,
        ssl_verify_identity=True,
        ssl_ca="/etc/ssl/cert.pem",
    )
elif env == "production":
    mysql_db = peewee.MySQLDatabase(
        "test",
        host=os.getenv("PROD_HOST"),
        port=4000,
        user=os.getenv("PROD_USER"),
        passwd=os.getenv("PROD_PASS"),
        ssl_verify_cert=True,
        ssl_verify_identity=True,
        ssl_ca="/etc/ssl/cert.pem",
    )
elif env == "local":
    mysql_db = peewee.MySQLDatabase(
        "test",
        host="127.0.0.1",
        port=4000,
        user="root",
        passwd="",
    )
else:
    exit_print_usage()


class UnsignedIntegerField(peewee.IntegerField):
    field_type = "int unsigned"


class Sample(peewee.Model):
    class Meta:
        database = mysql_db
        db_table = "sample"

    id = UnsignedIntegerField(
        # primary_key=True,
    )
    vec = VectorField(784)


def connect():
    print(
        f"+ Connecting to {mysql_db.connect_params['user']}@{mysql_db.connect_params['host']}..."
    )
    mysql_db.connect()


def run_load():
    connect()
    print("+ Creating tables...")
    mysql_db.create_tables([Sample])
    VectorAdaptor(mysql_db).create_vector_index(
        Sample.vec, tidb_vector.DistanceMetric.L2, skip_existing=True
    )

    with h5py.File("./fashion-mnist-784-euclidean.hdf5", "r") as data_file:
        data: numpy.ndarray = data_file["train"][()]
        data_with_id = [(idx, data[idx]) for idx in range(0, len(data))]

        for batch in peewee.chunked(data_with_id, 500):
            print(f"  - Insert PK: {batch[0][0]}")
            Sample.insert_many(batch, fields=[Sample.id, Sample.vec]).execute()

    print("+ Done")


def run_test():
    connect()
    print("+ Reading TIFLASH_INDEXES...")
    cursor = mysql_db.execute_sql(
        f"SELECT ROWS_STABLE_INDEXED, ROWS_STABLE_NOT_INDEXED, ROWS_DELTA_INDEXED, ROWS_DELTA_NOT_INDEXED FROM INFORMATION_SCHEMA.TIFLASH_INDEXES WHERE TIDB_TABLE='sample'"
    )
    print(
        tabulate.tabulate(
            cursor.fetchall(),
            headers=[
                "StableIndexed",
                "StableNotIndexed",
                "DeltaIndexed",
                "DeltaNotIndexed",
            ],
            tablefmt="psql",
        )
    )

    print("+ Querying...")
    with h5py.File("./fashion-mnist-784-euclidean.hdf5", "r") as data_file:
        test_rowid = 0

        query_row: numpy.ndarray = data_file["test"][test_rowid]
        groundtruth_results_set = set(data_file["neighbors"][test_rowid])

        # First query: select id, to calculate a recall rate.
        with mysql_db.execute_sql(
            "SELECT id FROM sample ORDER BY VEC_L2_Distance(vec, %s) LIMIT 100",
            (encode_vector(query_row),),
        ) as cursor:
            actual_results = cursor.fetchall()
            actual_results_set = set([int(row[0]) for row in actual_results])
            recall = (
                len(groundtruth_results_set & actual_results_set)
                / len(groundtruth_results_set)
                * 100
            )
            print(f"Recall@100: {recall:.2f}%")

        # Second query: Use EXPLAIN ANALYZE to check the (hot cache) performance.
        with mysql_db.execute_sql(
            "EXPLAIN ANALYZE SELECT id FROM sample ORDER BY VEC_L2_Distance(vec, %s) LIMIT 100",
            (encode_vector(query_row),),
        ) as cursor:
            print(tabulate.tabulate(cursor.fetchall(), tablefmt="psql"))


def run_clean():
    connect()
    print("+ Dropping tables...")
    mysql_db.drop_tables([Sample])
    print("+ Done")


def run_connect():
    subprocess.run(
        [
            "mysql",
            "-h",
            mysql_db.connect_params["host"],
            "-P",
            str(mysql_db.connect_params["port"]),
            "-u",
            mysql_db.connect_params["user"],
            f'-p{mysql_db.connect_params["passwd"]}',
        ],
        check=True,
    )


if action == "load":
    run_load()
elif action == "test":
    run_test()
elif action == "clean":
    run_clean()
elif action == "connect":
    run_connect()
else:
    exit_print_usage()

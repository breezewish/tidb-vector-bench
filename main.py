import os
import dotenv
import sys
import h5py
import numpy
import peewee
import pymysql
import tabulate


def exit_print_usage():
    print("Usage: python bench.py <local|dev|staging|production> <load|test>")
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


def format_vec(value: numpy.ndarray):
    return numpy.array2string(
        value,
        max_line_width=999999999999,
        separator=",",
        formatter={"float_kind": lambda x: "%.1f" % x},
    )


class VectorField(peewee.Field):
    field_type = "VECTOR(784)"

    def db_value(self, value):
        return format_vec(value)

    def python_value(self, value):
        return value


class UnsignedIntegerField(peewee.IntegerField):
    field_type = "int unsigned"


class Sample(peewee.Model):
    class Meta:
        database = mysql_db
        db_table = "sample"

    id = peewee.CharField(
        # primary_key=True,
    )
    vec = VectorField(
        constraints=[peewee.SQL("COMMENT 'hnsw(distance=cosine)'")],
    )


print(
    f"+ Connecting to {mysql_db.connect_params['user']}@{mysql_db.connect_params['host']}..."
)
mysql_db.connect()
print("+ Creating tables...")
mysql_db.create_tables([Sample])


def run_load():
    with h5py.File("./fashion-mnist-784-euclidean.hdf5", "r") as data_file:
        data: numpy.ndarray = data_file["train"][()]
        data_with_id = [(idx, data[idx]) for idx in range(0, len(data))]

        for batch in peewee.chunked(data_with_id, 500):
            print(f"  - Insert PK: {batch[0][0]}")
            Sample.insert_many(batch, fields=[Sample.id, Sample.vec]).execute()

    print("done")


def run_test():
    print("+ Reading TIFLASH_INDEXES...")
    cursor = mysql_db.execute_sql(
        f"SELECT ROWS_STABLE_INDEXED, ROWS_STABLE_NOT_INDEXED, ROWS_DELTA_NOT_INDEXED FROM INFORMATION_SCHEMA.TIFLASH_INDEXES WHERE TIDB_TABLE='sample'"
    )
    print(
        tabulate.tabulate(cursor.fetchall(), headers=["Indexed", "NotIndexed", "Delta"])
    )

    print("+ Querying...")
    with h5py.File("./fashion-mnist-784-euclidean.hdf5", "r") as data_file:
        query_row: numpy.ndarray = data_file["test"][0]
        cursor = mysql_db.execute_sql(
            "EXPLAIN ANALYZE SELECT id, VEC_Cosine_Distance(vec, %s) AS d FROM sample ORDER BY d LIMIT 10",
            (format_vec(query_row),),
        )
        print(tabulate.tabulate(cursor.fetchall()))


if action == "load":
    run_load()
elif action == "test":
    run_test()
else:
    exit_print_usage()

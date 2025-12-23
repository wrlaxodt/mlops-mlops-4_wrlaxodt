import os
import pandas as pd
from sqlalchemy import create_engine, text


def get_engine(db_name):
    engine = create_engine(
        f"mysql+mysqldb://"
        f"{os.environ.get('DB_USER')}:"
        f"{os.environ.get('DB_PASSWORD')}@"
        f"{os.environ.get('DB_HOST')}:"
        f"{os.environ.get('DB_PORT')}/"
        f"{db_name}"
    )
    return engine


def read_db(db_name, table_name, k=10):
    engine = get_engine(db_name)
    connect = engine.connect()

    result = connect.execute(
        text(
            f"select recommend_content_id "
            f"from {table_name} "
            f"order by `index` desc limit :k"
        ),
        {"k": k},
    )

    contents = [row[0] for row in result]
    connect.close()
    return contents

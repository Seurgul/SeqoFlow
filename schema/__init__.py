"""
Catalog for DB.
"""

import json
from pathlib import Path
from config.config import HOME


def get_table_schema(db_name, table_name):

    path = Path(HOME).joinpath(f"schema/{db_name}/{table_name}.json")
    with open(path, 'r') as f:
        table_schema = json.load(f)
    return table_schema

# import_manager.py
# Author: Thomas MINIER - MIT License 2017-2020
from importlib import import_module
from typing import Callable, Dict, List

from sage.database.db_connector import DatabaseConnector
from sage.asse.database.approx import DatabaseConnectorWithApproxSearch

BackendFactory = Callable[[Dict[str, str]], DatabaseConnector]
ApproxFactory = Callable[[Dict[str, str]], DatabaseConnectorWithApproxSearch]


def builtin_backends() -> Dict[str, BackendFactory]:
    """Load the built-in backends: HDT, PostgreSQL and MVCC-PostgreSQL.

    Returns: The HDT, PostgreSQL and MVCC-PostgreSQL backends, registered in a dict.
    """
    approx_data = [
        {
            'name': 'random',
            'path': 'sage.asse.database.random',
            'connector': 'RandomSearchConnector',
            'required': [
                'store',
            ]
        },
        {
            'name': 'dpr',
            'path': 'sage.asse.database.dpr',
            'connector': 'DensePassageRetrievalConnector',
            'required': [
            ]
        }
    ]
    exact_data = [
        # HDT backend (read-only)
        {
            'name': 'hdt-file',
            'path': 'sage.database.hdt.connector',
            'connector': 'HDTFileConnector',
            'required': [
                'file'
            ]
        },
        # PostgreSQL backend (optimised for read-only)
        {
            'name': 'postgres',
            'path': 'sage.database.postgres_backends.postgres.connector',
            'connector': 'DefaultPostgresConnector',
            'required': [
                'dbname',
                'user',
                'password'
            ]
        },
        # PostgreSQL backend with a catalog-based schema (optimised for read-only)
        {
            'name': 'postgres-catalog',
            'path': 'sage.database.postgres_backends.postgres_catalog.connector',
            'connector': 'CatalogPostgresConnector',
            'required': [
                'dbname',
                'user',
                'password'
            ]
        },
        # MVCC-PostgreSQL (read-write)
        {
            'name': 'postgres-mvcc',
            'path': 'sage.database.postgres_backends.postgres_mvcc.connector',
            'connector': 'MVCCPostgresConnector',
            'required': [
                'dbname',
                'user',
                'password'
            ]
        },
        # SQlite backend (optimised for read-only)
        {
            'name': 'sqlite',
            'path': 'sage.database.sqlite_backends.sqlite.connector',
            'connector': 'DefaultSQliteConnector',
            'required': [
                'database'
            ]
        },
        # SQlite backend (optimised for read-only)
        {
            'name': 'sqlite-catalog',
            'path': 'sage.database.sqlite_backends.sqlite_catalog.connector',
            'connector': 'CatalogSQliteConnector',
            'required': [
                'database'
            ]
        },
        # HBase backend
        {
            'name': 'hbase',
            'path': 'sage.database.hbase.connector',
            'connector': 'HBaseConnector',
            'required': [
                'thrift_host'
            ]
        }
    ]
    return {item['name']: import_backend(item['name'], item['path'], item['connector'], item['required']) for item in exact_data}, \
            {item['name']: import_approx(item['name'], item['path'], item['connector'], item['required']) for item in approx_data}


def import_backend(name: str, module_path: str, class_name: str, required_params: List[str]) -> BackendFactory:
    """Load a new database backend, defined by the user, adn get a factory function to build it.

    Args:
      * name: Name of the database backend.
      * module_path: Path to the python module which contains the backend implementation.
      * class_name: Name of the class that implements the backend. it must be a subclass of :class`sage.database.db_connector.DatabaseConnector`.
      * required_params: list of required configuration parameters for the backend.

    Returns:
      A factory function that build an instance of the new backend from a configuration object.

    Example:
      >>> name = "hdt-bis"
      >>> module_path = "sage.database.hdt.connector"
      >>> class_name = "HDTFileConnector"
      >>> params = [ "file" ]
      >>> factory = import_backend(name, module_path, class_name, params)
      >>> hdt_backend = factory({ "file": "/opt/data/hdt/dbpedia.hdt" })
    """
    # factory used to build new connector
    def __factory(params: Dict[str, str]) -> DatabaseConnector:
        # load module dynamically
        module = import_module(module_path)
        if not hasattr(module, class_name):
            raise RuntimeError(f"Connector class {class_name} not found in module {module_path}")
        connector = getattr(module, class_name)
        # check that all required params are present
        for key in required_params:
            if key not in params:
                raise SyntaxError(f"Missing required parameters for backend {name}. Expected to see {required_params}")
        return connector.from_config(params)
    return __factory

def import_approx(name: str, module_path: str, class_name: str, required_params: List[str]) -> ApproxFactory:
    def __factory(params: Dict[str, str], exact_backend: DatabaseConnector) -> DatabaseConnectorWithApproxSearch:
        module = import_module(module_path)
        if not hasattr(module, class_name):
            raise RuntimeError(f"Connector class {class_name} not found in module {module_path}")
        connector = getattr(module, class_name)
        for key in required_params:
            if key not in params:
                raise SyntaxError(f"Missing required parameters for backend {name}. Expected to see {required_params}")
        return connector.from_config(params, exact_backend)
    
    return __factory

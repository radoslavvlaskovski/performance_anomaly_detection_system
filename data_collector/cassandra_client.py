from threading import Event

import pandas as pd
from cassandra.auth import PlainTextAuthProvider
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement


class CassandraClient:

    def __init__(self, cassandra_host, cassandra_port, cassandra_user, cassandra_pass):
        self.cassandra_host = cassandra_host
        self.cassandra_port = cassandra_port
        self.cassandra_user = cassandra_user
        self.cassandra_pass = cassandra_pass

        self.auth_provider = self.create_auth_provider()
        self.cluster = self.register_cluster()

    def create_auth_provider(self):
        return PlainTextAuthProvider(username=self.cassandra_user, password=self.cassandra_pass)

    def register_cluster(self):
        return Cluster(contact_points=[self.cassandra_host], port=self.cassandra_port, auth_provider=self.auth_provider)

    def get_tables(self, db):
        sql_query = "SELECT * FROM system_schema.tables WHERE keyspace_name = '" + db + "';"
        session = self.cluster.connect(db)
        return session.execute(sql_query)

    def execute_query(self, db, query, output_file):
        session = self.cluster.connect(db)

        future = session.execute_async(SimpleStatement(query, fetch_size=4000))
        handler = PagedResultHandler(future, output_file)
        handler.finished_event.wait()
        if handler.error:
            raise handler.error


class PagedResultHandler(object):

    def __init__(self, future, output_file):
        self.error = None
        self.finished_event = Event()
        self.future = future
        self.future.add_callbacks(
            callback=self.handle_page,
            errback=self.handle_error)
        self.page = 0
        self.output_file = output_file
        self.output_dataframe = None

    def handle_page(self, rows):
        results = list()
        for row in rows:
            results.append(row)
        self.process_rows(results)
        if self.future.has_more_pages:
            self.page += 1
            self.future.start_fetching_next_page()
        else:
            self.output_dataframe.to_csv(self.output_file + ".csv", sep=",", index=False)
            self.finished_event.set()

    def process_rows(self, results):
        if self.output_dataframe is not None:
            self.output_dataframe = self.output_dataframe.append(pd.DataFrame(results))

        else:
            self.output_dataframe = pd.DataFrame(results)

    def handle_error(self, exc):
        self.error = exc
        self.finished_event.set()

    def get_results(self):
        print(len(self.results))
        return self.results

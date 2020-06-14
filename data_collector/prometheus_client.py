import requests
import os

class PrometheusClient:

    def __init__(self, prometheus_host, prometheus_port):
        self.prometheus_host = prometheus_host
        self.prometheus_port = prometheus_port

        self.prometheus_address = 'http://{0}:{1}/api/v1/query'.format(self.prometheus_host, self.prometheus_port)

    def execute_multiple_queries(self, queries):
        if len(queries) < 1:
            raise Exception("Need to specify at least one query.")

        out = {}
        for query in queries:
            name, app, values = self.execute_query(query)
            if app not in out.keys():
                out[app] = {}
            out[app][name] = values
        return out

    def execute_query(self, query):
        if query == "" or query is None:
            raise Exception("Query is empty")
        response = requests.get(self.prometheus_address, params={'query': query})
        print(response.json())
        return response.json()["data"]["result"][0]["metric"]['__name__'],\
               response.json()["data"]["result"][0]["metric"]['container_label_io_kubernetes_pod_name'],\
               response.json()["data"]["result"][0]["values"]

    def create_query(self, metric, filters=None, interval=None):
        if metric == "" or metric is None:
            raise Exception("You need to specify a metric")

        query = metric
        if filters is not None and len(filters) != 0:
            query += "{" + filters + "}"

        if interval is not None and len(interval) != 0:
            query += "[" + interval + "]"

        return query

    def queries_response_to_pandas_df(self, out, output_dir):
        import pandas as pd

        df = pd.DataFrame()

        for app in out.keys():
            time_stamps = [val[0] for val in out[app][list(out[app].keys())[0]]]
            time_stamps = pd.DataFrame({"time": time_stamps})
            df = pd.concat([df, time_stamps], axis=1)

            for metric in out[app].keys():
                values = out[app][metric]
                values = [val[1] for val in values]
                values = pd.DataFrame({metric+"_value": values})

                df = pd.concat([df, values], axis=1)

            output_file = os.path.join(output_dir, app + ".csv")
            df.to_csv(output_file, sep=",", index=False)


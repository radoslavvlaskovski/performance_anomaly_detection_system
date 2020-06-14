from cassandra_client import CassandraClient
from prometheus_client import PrometheusClient


def collect_traces(cassandra_client, output_file):
    sql_query = "SELECT trace_id, span_id, duration, flags, operation_name, parent_id, process, refs, start_time FROM {}.{};".format(
        cassandra_db, cassandra_table)
    cassandra_client.execute_query(cassandra_db, sql_query, output_file)


def collect_metrics(prometheus_client, metrics, pure_queries, interval=None, output_dir="out/"):
    for metric in metrics.keys():
        pure_queries.extend([prometheus_client.create_query(metric, filters, interval) for filters in metrics[metric].get_filters_list()])
    print(pure_queries)
    out = prometheus_client.execute_multiple_queries(pure_queries)
    prometheus_client.queries_response_to_pandas_df(out, output_dir)


def print_cassandra_tables(cassandra_agent, cassandra_db):
    for table in list(cassandra_agent.get_tables(cassandra_db)):
        print(table)


class PrometheusFilters:
    def __init__(self, filter_list):
        self.filters = filter_list

    def get_filters_list(self):
        return self.filters

if __name__ == "__main__":

    cassandra_user = "cassandra"
    cassandra_pass = "cassandra"
    cassandra_host = "127.0.0.1"
    cassandra_port = 9042

    cassandra_db = "jaeger_v1_dc1"
    cassandra_table = "traces"

    cassandra_client = CassandraClient(cassandra_host, cassandra_port, cassandra_user, cassandra_pass)

    print_cassandra_tables(cassandra_client, cassandra_db)
    collect_traces(cassandra_client, "out/120-220-no-anom/tracing_data")

    prometheus_host = "127.0.0.1"
    prometheus_port = "9090"

    prometheus_client = PrometheusClient(prometheus_host, prometheus_port)

    network_filters = PrometheusFilters(['container_label_app="currencyservice", interface="eth0"'])
    core_filters = PrometheusFilters(['container_label_app="currencyservice"'])
    cpu_usage_filters = PrometheusFilters(['image!="", container_name!="POD", container_label_io_kubernetes_pod_name="currencyservice-0", container_label_app=""'])


    metrics = {}
    metrics["container_network_receive_packets_total"] = network_filters
    metrics["container_network_receive_packets_dropped_total"] = network_filters
    metrics["container_network_transmit_packets_total"] = network_filters
    metrics["container_network_transmit_packets_dropped_total"] = network_filters
    metrics["container_network_receive_bytes_total"] = network_filters
    metrics["container_network_transmit_errors_total"] = network_filters
    metrics["container_memory_usage_bytes"] = cpu_usage_filters
    metrics["container_memory_working_set_bytes"] = cpu_usage_filters
    metrics["container_cpu_system_seconds_total"] = cpu_usage_filters
    metrics["container_cpu_usage_seconds_total"] = cpu_usage_filters

    pure_queries = [
    ]

    interval = "5h"

    collect_metrics(prometheus_client, metrics, pure_queries, interval, output_dir="out/120-220-no-anom")


## Performance Anomaly Detection System

This is the implementation of an Anomaly Detection System for Microservice architectures. 
It was implemented for the purpose of exploring different approaches to Anomaly Detection using Machine Learning
 for my Master Thesis at the Technical University of Berlin. 
 
### Architecture

The ADS has four main components: Data Collector, Data preprocessing, Machine Learning training and Threshold Functions.
The Data Collector supports collecting system metrics data from Prometheus and tracing data from Jaeger through a Cassandra DB.
The Machine Learning module can train different Neural Networks and scikit models. It also includes an implementation of RNN and FilterNet.

![Alt text](graphics/ADS_arch.png?raw=true "Anomaly Detection System Architecture")

### Installation

The system can be installed with the following commands:

```bash
pip install -r requirements.txt && python setup.py install
```

### License

The system is Open Source, licensed under the MIT License and free for commercial use.

Author: Radoslav Vlaskovski

Supervisors: Prof. Dr.-Ing. Stefan Tai &  Prof. Dr. habil. Odej Kao

Advisor: Dominik Ernst

Chair: Information  Systems Engineering, TU Berlin

![Alt text](graphics/tu.png?raw=true "")

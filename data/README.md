# Input schema

example_schema.csv contains headers only, with no internship observations. Supply an authorized hourly CSV separately as electricity_demand.csv. Required columns are Timestamp, Demand, Temperature and Humidity. Timestamp may contain the full timezone-naive hourly datetime; alternatively supply date-only Timestamp plus integer hour (0–23). Document the timezone and demand units. Do not commit confidential data. At least 168 warm-up hours plus sufficient fitting, validation, calibration and test history are required; the header file is not runnable research data.

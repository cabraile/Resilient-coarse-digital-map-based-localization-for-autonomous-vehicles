import numpy as np

def print_statistics(list_error, list_benchmark_duration, list_timestamp, time_localized):
    avg_error = np.average(list_error["error"])
    std_error = np.std(list_error["error"])
    avg_time_pred = np.average(list_benchmark_duration["prediction"])
    avg_time_updt = np.average(list_benchmark_duration["update"])
    avg_time_detect = np.average(list_benchmark_duration["detection"])
    total_ellapsed_time = (list_timestamp[-1] - list_timestamp[0]) * 1e-9
    print("---------------------------------------")
    print("Error-related run stats:")
    print("> avg_error = ", avg_error)
    print("> std_error = ", std_error)
    print("> time_localized (in seconds) = ", time_localized)
    print("> total_ellapsed_time (in seconds) = ", total_ellapsed_time)
    print("> prop_time_localized  = ", time_localized/total_ellapsed_time)
    print("Time-related run stats:")
    print("> avg_time_pred = ", avg_time_pred)
    print("> avg_time_updt = ", avg_time_updt)
    print("> avg_time_detect = ", avg_time_detect)
    print("---------------------------------------")
    return
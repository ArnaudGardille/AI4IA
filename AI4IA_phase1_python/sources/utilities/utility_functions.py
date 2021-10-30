import signal
import pandas as pd


def set_alarm_clock(timeout):
    """
    Raise TimeoutError if time > timeout.
    Remember to deactivate alarm if not required - use timeout=-1
    :param timeout: in seconds.
    :return:
    """
    def signal_handler(signum, frame):
        raise TimeoutError("Timeout")

    signal.signal(signal.SIGALRM, signal_handler)
    if timeout > 0:
        signal.alarm(timeout)
    elif timeout<0:
        signal.alarm(10**10)

def load_data_json(data_path):
    df = pd.read_json(data_path)
    xs = []
    ys = []
    for col in df.columns:
        if '_in' in col:
            xs.append(col)
        if '_out' in col:
            ys.append(col)

    xs = [df[['time',col]].to_numpy(dtype=float) for col in xs]
    ys = [df[col].to_numpy(dtype=float).reshape(-1,1) for col in ys]
    return xs, ys

def load_data_csv(data_path,name_time="Time",
                            name_input="input",
                            names_outputs = ["output"+str(k) for k in range(1,6)],
                            Ndecim = 1
                 ):
    
    df = pd.read_csv(data_path,sep=',',index_col=False)
    
    return df[name_time].to_numpy()[::Ndecim],df[name_input].to_numpy()[::Ndecim],df[names_outputs].to_numpy()[::Ndecim]
import pandas
import cv2

class DataManager:

    def __init__(self,msgs):
        """
        Caution: this piece of code assumes each sensor measurement is performed once per timestamp.

        @param msgs: dict of strings. Contains the path of the CSV for each of the sensors.
        """

        self.data = {}

        # Load odometer measurements to a pandas DataFrame - with timestamp
        self.data["odometer"] = pandas.read_csv(msgs["odometer"], sep=",", dtype={"timestamp" : str}, index_col="timestamp")

        # Load image path to a pandas Dataframe - with respective timestamps
        self.data["image"] =  pandas.read_csv(msgs["image"], sep=",", dtype={"timestamp" : str}, index_col="timestamp")

        # Load groundtruth values to a pandas Dataframe - with respective timestamps
        self.data["groundtruth"] = pandas.read_csv(msgs["groundtruth"], sep=",", dtype={"timestamp" : str}, index_col="timestamp")

        # List all timestamps from received messages during the run
        timestamps = \
            self.data["odometer"].index.tolist() + \
            self.data["image"].index.tolist() + \
            self.data["groundtruth"].index.tolist()

        # Load compass measurements to a pandas DataFrame - with respective timestamps
        self.data["compass"] = {}
        if("compass" in msgs):
            self.data["compass"] = pandas.read_csv(msgs["compass"], sep=",", index_col="timestamp")
            timestamps += self.data["compass"].index.tolist()

        timestamps = set(timestamps)    # Remove duplicates
        timestamps = list(timestamps)   # Required for sort
        timestamps.sort()               # Timestamps in chronological order
        self.timestamps = timestamps
        self.time_idx = 0
        return

    def next(self):
        """
            Retrieves the data for the next time step.
        """
        if(not self.hasNext()):
            raise Exception("Error: no new data to be accessed")
        ret = {}
        ts = self.timestamps[self.time_idx]
        ret["timestamp"] = ts
        for data_type in ["odometer", "image", "groundtruth", "compass"]:
            if(ts in self.data[data_type].index):
                if(data_type != "image"):
                    ret[data_type] = self.data[data_type].loc[ts]
                else:
                    path = self.data[data_type].loc[ts]["path"]
                    try:
                        ret[data_type] = cv2.imread(path)[:,:,::-1]
                    except:
                        print("\nError: could not load an image from the dataset. Path: '{}'.".format(path))                        
            else:
                ret[data_type] = None
        self.time_idx+=1
        return ret

    def hasNext(self):
        """
        Checks whether there is still data to be read or if all of the data was already read.

        @return True if there is still data to be read. False otherwise.
        """
        if(self.time_idx < len(self.timestamps)):
            return True
        return False


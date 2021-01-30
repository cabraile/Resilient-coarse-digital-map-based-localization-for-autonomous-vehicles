import cv2
import yaml
import utm
from datetime import datetime

class Landmark :

    def __init__(self, uid=None, coordinates = None, heading = None, images = [], name=""):
        """
        @param coordinates: tuple.
            The tuple containing the latitude and longitude of the node.
        @param images: list.
            A list of the photos taken of the landmark.
        """
        self.id =  uid
        self.__images__     = images[:]
        self.__point__      = None
        self.__utm_zone__   = None
        self.__heading__    = heading
        self.date           = None # type datetime.datetime
        self.name           = name
        self.__image_features__ = []
        if(coordinates is not None):
            self.set_coordinate(coordinates[0], coordinates[1])
        return

    def get_position(self):
        return self.__point__

    def set_coordinate(self, lat, lon):
        """
        Given the latitude and the longitude, store it as a (easting,northing) 2D point
        """
        self.__coordinates__ = (lat, lon)
        ret = utm.from_latlon(lat, lon)
        self.__point__ = (ret[0], ret[1])
        self.__utm_zone__ = str(ret[2]) + str(ret[3])
        return

    def add_image(self,img):
        self.__images__.append(img)
        return

    def images(self):
        return self.__images__[:]

    def set_image_features(self, features):
        self.__image_features__.append(features.ravel()[:])
        return

    def features(self):
        return self.__image_features__[:]

    def from_yaml(self,path):
        """
        Prototype function. OSM retrieves XML files, so this function is only
        going to be used in this prototype. Docstring is the same as 'from_xml'

        Keys:
            id: the id of the node
            lat: the latitude
            lon: the longitude
            timestamp: date in str format of the last edit
            images: a list of image path
            name: a name for that particular node (OPTIONAL)
        Other keys include tags from OSM
        """
        yaml_dict = None
        with open(path) as f:
            try:
                yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
            except:
                print("Could not load file {}".format(path))
        self.set_coordinate(yaml_dict["lat"], yaml_dict["lon"])
        self.id = yaml_dict["id"]
        if("name" in yaml_dict):
            self.name = yaml_dict["name"]
        date_str = yaml_dict["timestamp"].split("T")[0]
        self.date = datetime.strptime(date_str, "%Y-%m-%d")
        path = yaml_dict["path"]
        self.__heading__ = yaml_dict["heading"]
        try:
            self.__images__.append( cv2.imread(path)[:,:,::-1] ) # Stores as RGB images
        except:
            print("Could not load image {}".format(path))
        return

    def from_xml(self,path):
        """
        Load all the information of the landmark. From lat-lon coordinates 
        to each image.

        Tags: TODO
        """
        pass

    def __str__ (self):
        heading = None
        if(self.__heading__ is not None):
            heading = 180.0 * (self.__heading__ / 3.1416)
        txt = "Node {}(ID {}) \nPosition: ({}, {})m, Zone: {}, Heading = {}degrees\nDate: {}\n".format(
            self.name, self.id, self.__point__[0], self.__point__[1], self.__utm_zone__, heading, self.date
        )
        return txt


if __name__ == "__main__":
    from matplotlib.pyplot import *
    lm = Landmark()
    lm.from_yaml("/mnt/58C297EC07BBE4EF/Workspace/Projects/Human-made-Map-based-Localization-Proto/config/.track_sao_carlos/landmarks/files/seo_gera.yaml")
    print(lm)
    imshow(lm.__images__[0])
    show()
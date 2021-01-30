from numpy import arctan2, linalg, array
import utm

class Way:

    def __init__(self, way_uid, street_id, coord_init, coord_end, maxspeed, oneway):
        """
        @param way_uid: int. Each way has its own, unique, identifier.
        @param street_id: hasheable. The identifier of the street from which the way belongs.
        @param coord_init: tuple (2D). The latitude and the longitude of the init position of the way
        @param coord_end: tuple (2D). The latitude and the longitude of the end position of the way
        @param maxspeed: int. Defines the speed limit in that way segment.
        @param oneway: bool. Defines whether the way belongs to a oneway or a twoway street segment.
        @param is_utm: bool. True if the provided points are in UTM coordinates. False if in latitude longitude
        """
        self.__uid__ = way_uid
        self.__coord_init__ = coord_init
        self.__coord_end__ = coord_end
        _ret = utm.from_latlon(coord_init[0], coord_init[1])
        self.__p_init__ = (_ret[0], _ret[1])
        self.__p_init_utm_zone_info__ = { "number" : _ret[2], "letter" : _ret[3] }
        _ret = utm.from_latlon(coord_end[0], coord_end[1])
        self.__p_end__  = (_ret[0], _ret[1])
        self.__p_end_utm_zone_info__ = { "number" : _ret[2], "letter" : _ret[3] }
        self.__street_id__ = street_id
        self.__maxspeed__ = maxspeed
        self.__oneway__ = oneway

        self.__angle__ = arctan2(
            self.__p_end__[1] - self.__p_init__[1], 
            self.__p_end__[0] - self.__p_init__[0]
        )
        self.__length__ = linalg.norm(array(self.__p_end__) - array(self.__p_init__))
        return
    
    def uid(self):
        """
        Getter. Retrieve the unique id of the way.
        """
        return self.__uid__

    def maxspeed(self):
        return self.__maxspeed__

    def p_init(self):
        """
        Getter. Retrieve the initial point from the way segment.
        """
        return self.__p_init__
    
    def p_end(self):
        """
        Getter. Retrieve the final point from the way segment.
        """
        return self.__p_end__

    def street_id(self):
        """
        Getter. Retrieve the identifier of the street from which the way belongs.
        """
        return self.__street_id__

    def orientation(self):
        """
        Getter. Retrieve the orientation of the way (from p_init to p_end) in rads.
        """
        return self.__angle__
    
    def __str__(self):
        return "{} (UID {}): maxspeed = {} | oneway = {}".format(self.__street_id__, self.__uid__, self.__maxspeed__, self.__oneway__)

    def length(self):
        """
        Getter. Retrieve the length of the way (from p_init to p_end). 
        The returned length is in the unit system of the points p_init and p_end.
        """
        return self.__length__
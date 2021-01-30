import keras
import numpy as np
import cv2

from modules.map.landmark import Landmark

DEFAULT_ROWS = 128
DEFAULT_COLS = 128
DEFAULT_DIMS = (DEFAULT_COLS, DEFAULT_ROWS)

class HolisticFeatureMatcher:

    def __init__(self, landmarks, model_name = "vgg16", threshold = 0.8):
        """
        @param landmarks: list of map.Landmark. 
            All the Landmarks contained in the map must be in this list
        @param model_name: string.
            The name of the model used for feature extraction.
            Possible names: "vgg16", "vgg19", "inception_v3", "handcrafted".
        @param threshold: float.
            Threshold for the similarity measured during matching. 
            Similarity values below the threshold are not considered as a match.
        """
        self.__image_target_size__ = DEFAULT_DIMS
        self.landmarks = landmarks
        self.model = None
        self.threshold = threshold
        self.__load_feature_extractor__(model_name)
        self.__compute_features__()
        return

    def __prepare_image__(self, img):
        """
        Crop, resize and add extra dimension to the input image.

        @param img: ndarray of rank 3.
            The RGB image to be processed.

        @return The prepared image.
        """
        # Square image
        if(img.shape[0] != img.shape[1]):
            min_dim_size    = min(img.shape[0], img.shape[1])
            center_row      = int(img.shape[0]/2)
            center_col      = int(img.shape[1]/2)
            half            = int(min_dim_size/2)
            img_square      = img[center_row-half: center_row+half, center_col-half : center_col + half]
        else:
            img_square      = img[:,:,:]
        
        # Resize to the default size
        image_resized   = cv2.resize(img_square, dsize=DEFAULT_DIMS, interpolation=cv2.INTER_CUBIC)
        img_data        = np.expand_dims(image_resized, axis=0)
        img_data        = self.preprocess_function(img_data)

        return img_data

    def extract_features(self, img):
        """
        Holistic feature extraction. Returns the output of the feature extraction model.

        @param img: ndarray of rank 3.
            The RGB image from which the features are going to be computed.
        @return The flattened feature array
        """
        image_prep = self.__prepare_image__(img)
        features = self.model.predict(image_prep)
        return features.ravel()

    def __load_feature_extractor__(self, model_name):
        if(model_name == "vgg16"):
            self.model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(DEFAULT_ROWS,DEFAULT_COLS,3))
            self.preprocess_function = keras.applications.vgg16.preprocess_input
        elif(model_name == "vgg19"):
            self.model = keras.applications.vgg19.VGG19(weights='imagenet', include_top=False, input_shape=(DEFAULT_ROWS,DEFAULT_COLS,3))
            self.preprocess_function = keras.applications.vgg19.preprocess_input
        elif(model_name == "inception"):
            self.model = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(DEFAULT_ROWS,DEFAULT_COLS,3))
            self.preprocess_function = keras.applications.inception_v3.preprocess_input
        elif(model_name == "resnet"):
            self.model = keras.applications.ResNet152V2(weights='imagenet', include_top=False, input_shape=(DEFAULT_ROWS,DEFAULT_COLS,3))
            self.preprocess_function = keras.applications.resnet_v2.preprocess_input
        elif(model_name == "handcrafted"):
            self.model = ImageDescriptor()
            self.preprocess_function = lambda img : img # identity function - no preprocess required
        return 

    def __compute_features__(self):
        """
        Compute and store the dataset features.
        """
        # Load and compute features
        for idx in range(0,len(self.landmarks)):
            image = self.landmarks[idx].images()[0]
            feature_vector = self.extract_features(image).ravel()
            self.landmarks[idx].set_image_features(feature_vector/np.linalg.norm(feature_vector))
        return

    def match(self, query_image):
        """
        @return Landmark. The landmark with the higher feature similarity to the input.
        @return float. The duration of the matching process.
        """
        count = 0
        max_similarity = None
        max_similarity_idx = None

        # Compute feature vector
        raw_feat_query = self.extract_features(query_image).ravel()
        
        # Process
        norm = np.linalg.norm(raw_feat_query)
        feat_query = raw_feat_query/norm

        for idx in range(len(self.landmarks)):
            # Loading entry
            # -------------------------------------
            candidate = self.landmarks[idx]
            loaded_feature = candidate.features()[0]

            # Comparison by similarity (cosine distance - both features are normalized)
            # --------------------------------------
            dist_cos = loaded_feature.dot(feat_query)
            if(max_similarity is None):
                max_similarity = dist_cos
                max_similarity_idx = idx
            else:
                if(max_similarity < dist_cos):
                    max_similarity = dist_cos
                    max_similarity_idx = idx

        if(max_similarity is None):
            return None
        if(max_similarity < self.threshold):
            return None
        print("\n[Matching Landmarks] Match similarity: ", max_similarity)
        match = self.landmarks[max_similarity_idx]
        return match
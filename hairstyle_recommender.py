from config import FACE_SHAPE_CRITERIA, HAIRSTYLE_RECOMMENDATIONS

class HairstyleRecommender:
    def __init__(self):
        self.face_shapes = FACE_SHAPE_CRITERIA
        self.recommendations = HAIRSTYLE_RECOMMENDATIONS
        
    def get_recommendations(self, face_shape):
        """
        Get hairstyle recommendations based on face shape
        
        Args:
            face_shape (str): The determined face shape
            
        Returns:
            tuple: (face_shape_description, recommendations)
        """
        if face_shape not in self.face_shapes:
            # Default to oval face if shape not recognized
            face_shape = "OVAL"
            
        face_shape_description = self.face_shapes[face_shape]["description"]
        recommended_hairstyles = self.recommendations[face_shape]
        
        return face_shape_description, recommended_hairstyles

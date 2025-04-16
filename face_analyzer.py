import cv2
import numpy as np
import mediapipe as mp
from config import FACE_SHAPE_CRITERIA

class FaceAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5
        )
        
    def analyze_face_shape(self, image_data):
        """
        Analyze face shape using facial landmarks from MediaPipe
        
        Args:
            image_data: Image bytes
            
        Returns:
            tuple: (face_shape, visualization_image, measurements)
        """
        try:
            # Convert image bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image to get facial landmarks
            results = self.face_mesh.process(image_rgb)
            
            # Check if a face was detected
            if not results.multi_face_landmarks:
                return None, None, None
                
            # Get the first face detected
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract key measurements from landmarks
            height, width, _ = image.shape
            landmarks = []
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * width), int(landmark.y * height)
                landmarks.append((x, y))
                
            # Draw landmarks on the face for visualization
            vis_image = image.copy()
            for landmark in landmarks:
                cv2.circle(vis_image, landmark, 1, (0, 255, 0), -1)
                
            # Connect landmarks to show face mesh
            connections = self.mp_face_mesh.FACEMESH_TESSELATION
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    cv2.line(vis_image, landmarks[start_idx], landmarks[end_idx], (0, 255, 0), 1)
            
            # Get key facial measurements
            # Forehead width (between temples)
            forehead_left = landmarks[67]
            forehead_right = landmarks[296]
            forehead_width = self._calculate_distance(forehead_left, forehead_right)
            
            # Jawline width
            jaw_left = landmarks[172]
            jaw_right = landmarks[397]
            jawline_width = self._calculate_distance(jaw_left, jaw_right)
            
            # Cheekbone width
            cheek_left = landmarks[123]
            cheek_right = landmarks[352]
            cheekbone_width = self._calculate_distance(cheek_left, cheek_right)
            
            # Face length (from chin to forehead)
            forehead_middle = landmarks[10]
            chin_point = landmarks[152]
            face_length = self._calculate_distance(forehead_middle, chin_point)
            
            # Calculate ratios
            width_to_length_ratio = cheekbone_width / face_length
            forehead_to_jawline_ratio = forehead_width / jawline_width
            cheekbone_to_jawline_ratio = cheekbone_width / jawline_width
            
            # Determine face shape based on ratios
            face_shape = self._determine_face_shape(
                width_to_length_ratio,
                forehead_to_jawline_ratio,
                cheekbone_to_jawline_ratio
            )
            
            # Measurements for debugging and visualization
            measurements = {
                "width_to_length_ratio": width_to_length_ratio,
                "forehead_to_jawline_ratio": forehead_to_jawline_ratio,
                "cheekbone_to_jawline_ratio": cheekbone_to_jawline_ratio,
            }
            
            # Encode the visualization image
            _, buffer = cv2.imencode('.jpg', vis_image)
            vis_image_bytes = buffer.tobytes()
            
            return face_shape, vis_image_bytes, measurements
            
        except Exception as e:
            print(f"Error in analyzing face: {e}")
            return None, None, None
            
    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
        
    def _determine_face_shape(self, width_to_length_ratio, forehead_to_jawline_ratio, cheekbone_to_jawline_ratio):
        """
        Determine face shape based on facial measurements and ratios
        """
        # Check each face shape criteria
        for shape, criteria in FACE_SHAPE_CRITERIA.items():
            # Check if all ratios match the criteria for this shape
            width_length_range = criteria["ratio_width_to_length"]
            forehead_jaw_range = criteria["forehead_to_jawline_ratio"]
            cheek_jaw_range = criteria["cheekbone_to_jawline_ratio"]
            
            if (width_length_range[0] <= width_to_length_ratio <= width_length_range[1] and
                forehead_jaw_range[0] <= forehead_to_jawline_ratio <= forehead_jaw_range[1] and
                cheek_jaw_range[0] <= cheekbone_to_jawline_ratio <= cheek_jaw_range[1]):
                return shape
                
        # Default to OVAL if no match found
        return "OVAL"

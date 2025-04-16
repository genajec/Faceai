import logging
import os
import tempfile
import io
import telebot

from config import TELEGRAM_API_TOKEN, BOT_MESSAGES
from face_analyzer import FaceAnalyzer
from hairstyle_recommender import HairstyleRecommender

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class FaceShapeBot:
    def __init__(self):
        self.bot = telebot.TeleBot(TELEGRAM_API_TOKEN)
        self.face_analyzer = FaceAnalyzer()
        self.hairstyle_recommender = HairstyleRecommender()
        
        # Регистрация обработчиков сообщений
        @self.bot.message_handler(commands=['start'])
        def handle_start(message):
            self.start(message)
            
        @self.bot.message_handler(commands=['help'])
        def handle_help(message):
            self.help_command(message)
            
        @self.bot.message_handler(content_types=['photo'])
        def handle_photo(message):
            self.process_photo(message)
            
        @self.bot.message_handler(content_types=['text'])
        def handle_text(message):
            # Проверяем, не является ли это командой
            if message.text.startswith('/'):
                return
            self.handle_message(message)
        
    def start(self, message):
        """Send a message when the command /start is issued."""
        chat_id = message.chat.id
        self.bot.send_message(chat_id, BOT_MESSAGES["start"])

    def help_command(self, message):
        """Send a message when the command /help is issued."""
        chat_id = message.chat.id
        self.bot.send_message(chat_id, BOT_MESSAGES["help"])

    def process_photo(self, message):
        """Process the user photo and send face shape analysis with recommendations."""
        chat_id = None
        try:
            chat_id = message.chat.id
            
            # Send processing message
            self.bot.send_message(chat_id, BOT_MESSAGES["processing"])
            
            # Get the largest photo (best quality)
            photos = message.photo
            if not photos:
                self.bot.send_message(chat_id, BOT_MESSAGES["no_face"])
                return
                
            photo = photos[-1]  # Get largest photo
            
            # Download the photo
            file_info = self.bot.get_file(photo.file_id)
            downloaded = self.bot.download_file(file_info.file_path)
            
            # Analyze the face
            face_shape, vis_image_bytes, measurements = self.face_analyzer.analyze_face_shape(downloaded)
            
            if face_shape is None:
                self.bot.send_message(chat_id, BOT_MESSAGES["no_face"])
                return
                
            # Get hairstyle recommendations
            face_shape_description, recommendations = self.hairstyle_recommender.get_recommendations(face_shape)
            
            # Format the message
            result_message = [
                f"✅ Анализ завершен!",
                f"",
                f"📊 Форма твоего лица: {face_shape_description}",
                f"",
                "💇 Рекомендации по стрижкам:"
            ]
            result_message.extend(recommendations)
            
            # Add some measurements for context (optional)
            if measurements:
                result_message.append("")
                result_message.append("📏 Измерения (технические данные):")
                for key, value in measurements.items():
                    result_message.append(f"- {key}: {value:.2f}")
                    
            # Send the visualization image with facial landmarks
            if vis_image_bytes:
                vis_image_io = io.BytesIO(vis_image_bytes)
                vis_image_io.name = 'face_analysis.jpg'
                self.bot.send_photo(
                    chat_id,
                    vis_image_io,
                    caption="Анализ лицевых точек"
                )
                
            # Send the recommendations
            self.bot.send_message(chat_id, "\n".join(result_message))
            
        except Exception as e:
            logger.error(f"Error processing photo: {e}")
            try:
                if chat_id:
                    self.bot.send_message(chat_id, BOT_MESSAGES["error"])
                else:
                    logger.error("Chat ID is None, can't send error message")
            except:
                logger.error("Failed to send error message to user")

    def handle_message(self, message):
        """Handle non-photo messages."""
        chat_id = message.chat.id
        self.bot.send_message(chat_id, BOT_MESSAGES["non_photo"])

    def run(self):
        """Run the bot."""
        logger.info("Starting bot...")
        
        # Start polling
        self.bot.polling(none_stop=True)

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch
from PIL import Image
import io
import base64
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import warnings
import os
from werkzeug.serving import WSGIRequestHandler

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__, template_folder='templates')
CORS(app)


class ModelManager:
    _instance = None
    _model = None
    _processor = None
    _device = None
    _initialization_status = "not_started"

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if ModelManager._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ModelManager._instance = self

    def get_status(self):
        return self._initialization_status

    def load_model(self):
        try:
            if self._model is None:
                self._initialization_status = "loading"
                print("Loading model...")

                # Set up cache directory
                cache_dir = os.path.join(os.getcwd(), "model_cache")
                os.makedirs(cache_dir, exist_ok=True)
                os.environ['TRANSFORMERS_CACHE'] = cache_dir

                # Load processor and model
                self._processor = AutoProcessor.from_pretrained(
                    "Salesforce/blip2-opt-2.7b",
                    cache_dir=cache_dir
                )

                self._model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b",
                    torch_dtype=torch.float16,
                    cache_dir=cache_dir
                )

                # Set device
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._model.to(self._device)

                self._initialization_status = "ready"
                print(f"Model loaded successfully on {self._device}!")
                return True

        except Exception as e:
            self._initialization_status = "failed"
            print(f"Error loading model: {str(e)}")
            return False

    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model

    @property
    def processor(self):
        if self._processor is None:
            self.load_model()
        return self._processor

    @property
    def device(self):
        if self._device is None:
            self.load_model()
        return self._device


# Initialize model manager
model_manager = ModelManager.get_instance()


def base64_to_image(base64_string):
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")


@app.route('/status')
def get_status():
    return jsonify({
        'status': model_manager.get_status(),
        'device': model_manager.device if model_manager._device else 'not_initialized'
    })


@app.route('/analyze', methods=['POST'])
def analyze_image():
    if model_manager.get_status() != "ready":
        return jsonify({
            'error': 'Model is not ready yet',
            'status': model_manager.get_status()
        }), 503

    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        image_data = data.get('image')
        question = data.get('question')

        if not image_data:
            return jsonify({'error': 'No image data provided'}), 400
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        # Process image
        image = base64_to_image(image_data)

        # Prepare input
        prompt = f"Question: {question} Answer:"
        inputs = model_manager.processor(
            image,
            text=prompt,
            return_tensors="pt"
        ).to(model_manager.device)

        # Generate response
        with torch.no_grad():
            generated_ids = model_manager.model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=3,
                temperature=0.5,
                top_p=0.8,
                length_penalty=1.2
            )

        generated_text = model_manager.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0].strip()

        return jsonify({
            'answer': generated_text,
            'status': 'success'
        })

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500


@app.route('/')
def home():
    return render_template('index.html')


def start_server():
    # Initialize model in the main thread
    print("Initializing model...")
    success = model_manager.load_model()

    if success:
        # Use threading server instead of werkzeug default
        WSGIRequestHandler.protocol_version = "HTTP/1.1"
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)
    else:
        print("Failed to initialize model. Please check the error messages above.")


if __name__ == '__main__':
    start_server()
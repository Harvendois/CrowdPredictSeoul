from flask import Flask, jsonify, request, json
from flask_cors import CORS, cross_origin
import PIL.Image as Image
from torchvision import datasets, transforms
import sys
# sys.path.append('D:/CJH/CrowdCount/CSRNet-pytorch')
# from model import CSRNet
sys.path.append('D:\CJH\CrowdPredict')
from predict import predict_population
import torch 
import io
import h5py 

app = Flask(__name__)
CORS(app)

# transform=transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# model = CSRNet()
# model = model.cuda()
# checkpoint = torch.load('D:/CJH/CrowdCount/CSRNet-pytorch/0model_best.pth.tar')
# model.load_state_dict(checkpoint['state_dict'])

@app.route('/')
def index():
    return "Hello, World!"

@app.route('/model', methods=['POST'])
def process_data():
    data = request.get_json()  # JSON 데이터 가져오기
    # 데이터 처리 로직 수행
    processed_data = {}  # 예시: 데이터를 처리한 결과를 저장하는 변수
    image=data["image"]
    # print(image)
    processed_data = {"answer" : "150"}
    response = jsonify(processed_data)  # 처리된 데이터를 JSON 형태로 변환하여 응답 생성
    return response

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    predicted_hours = data['predicted_hours']
    all_predictions = predict_population(predicted_hours)
    return jsonify(all_predictions)

# @app.route('/crowdcount', methods=['POST'])
# def crowd_count():
#     # Check if a file is included in the request
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'})

#     file = request.files['file']
#     # Read the image file to PIL, transform and convert to tensor
#     image = Image.open(io.BytesIO(file.read())).convert('RGB')
#     image = transform(image).cuda()

#     # Pass the file to your deep learning model for predictions
#     output = model(image.unsqueeze(0))
#     prediction = int(output.detach().cpu().sum().numpy())

#     # Create a JSON response with the predicted value
#     response = {
#         'prediction': prediction,
#         'mae': 69.75169009952755
#     }

#     # Return the response as JSON
#     return jsonify(response)

if __name__ == '__main__':
    app.run('192.168.0.122', port=5000, debug=True)

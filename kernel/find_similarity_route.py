import os
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from lunetkernel import comparator

app = Flask(__name__)

@app.route(
    '/find-similarity',
    methods=['POST']
)
@cross_origin(allow_headers=['Content-Type'])
def find_similarity_route():
    '''
        This view has CORS enabled for all domains, and allows browsers
        to send the Content-Type header, allowing cross domain AJAX POST
        requests.
        $ http post  http://0.0.0.0:5000/find-similarity
        HTTP/1.0 200 OK
        Access-Control-Allow-Headers: Content-Type
        Access-Control-Allow-Origin: *
        {
            "success": true
        }
    '''
    img1 = request.json['img1']
    img2 = request.json['img2']
    sim = comparator(img1, img2)
    return jsonify({'similarity': sim}), 200

if __name__ == "__main__":
    app.secret_key = os.urandom(24)
    app.run()
    #app.run(debug=True,host="0.0.0.0",use_reloader=False)

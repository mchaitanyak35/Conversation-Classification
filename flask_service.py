from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from classification import predict


app = Flask(__name__)
api = Api(app)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class Predict(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']

        prediction = predict(user_query)

        output = {'user_query': user_query, 'prediction': prediction}

        return output


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(Predict, '/predict')


if __name__ == '__main__':
    app.run(debug=True)

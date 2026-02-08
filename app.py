from flask import Flask, request, jsonify, render_template
from recommender import HybridRecommender

app = Flask(__name__)
model = HybridRecommender('data/products.csv', 'data/interactions.csv')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend-ui')
def recommend_ui():
    product_id = int(request.args.get('product_id'))
    recommendations = model.recommend(product_id)

    error = None
    if not recommendations:
        error = "Invalid Product ID or no recommendations found."

    return render_template(
        'index.html',
        recommendations=recommendations,
        error=error
    )



@app.route('/recommend')
def recommend():
    product_id = int(request.args.get('product_id'))
    return jsonify(model.recommend(product_id))


if __name__ == '__main__':
    app.run(debug=True)

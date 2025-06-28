from flask import render_template, redirect, url_for
from predictor_app.forms import PredictionForm
from predictor_app import app
from model import predict_price

@app.route('/', methods=['GET', 'POST'])
def home():
    form = PredictionForm()
    if form.validate_on_submit():
        current_price = form.current_price.data
        area = form.area.data
        quality = form.quality.data
        predicted_price = predict_price(current_price, area, quality)
        return render_template('results.html', predicted_price=predicted_price)
    return render_template('index.html', form=form)

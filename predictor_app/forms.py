from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField
from wtforms.validators import DataRequired, NumberRange

class PredictionForm(FlaskForm):
    current_price = FloatField('Current Price (k)', validators=[DataRequired(), NumberRange(min=0, message="Price must be a positive number")])
    area = FloatField('Area (sqft)', validators=[DataRequired(), NumberRange(min=0, message="Area must be a positive number")])
    quality = FloatField('Quality (1-10)', validators=[DataRequired(), NumberRange(min=1, max=10, message="Quality must be between 1 and 10")])
    submit = SubmitField('Predict Price')
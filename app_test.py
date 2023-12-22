from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('linreg_model.pkl', 'rb'))

@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')

scaler = StandardScaler()

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Age = float(request.form['AGE'])
        Leaves_used = float(request.form['LEAVES USED'])
        Ratings = float(request.form['RATINGS'])
        Past_Exp = float(request.form['PAST EXP'])
        Yoe = float(request.form['YOE_ORG'])
        #SEX_Fem = 1 if request.form['SEX_F'] == 'F' else 0

        # Standardizing selected columns
        standardized_features = scaler.fit_transform([[Age, Leaves_used, Ratings, Past_Exp, Yoe]])

        # Other categorical features
        DESIGNATION = request.form['DESIGNATION_Analyst']
        UNIT = request.form['UNIT_Finance']
        SEX = request.form['SEX_F']

        #Mapping sex to binary features
        SEX_VALUES = {
            'F': [1,0],
            'M': [0,1]
            
        }
        SEX_VALUES.get(SEX, [1,0]) #Defaulting to Female if not found


        # Mapping designations to binary features
        DESIGNATION_VALUES = {
            'Analyst': [1, 0, 0, 0, 0, 0],
            'Associate': [0, 1, 0, 0, 0, 0],
            'Director': [0, 0, 1, 0, 0, 0],
            'Manager': [0, 0, 0, 1, 0, 0],
            'Senior Analyst': [0, 0, 0, 0, 1, 0],
            'Senior Manager': [0, 0, 0, 0, 0, 1]
        }
        DESIGNATION_VALUES.get(DESIGNATION, [1, 0, 0, 0, 0, 0])  # Defaulting to Analyst if not found

        # Mapping UNIT to binary features
        UNIT_VALUES = {
            'Finance': [1, 0, 0, 0, 0, 0],
            'IT': [0, 1, 0, 0, 0, 0],
            'Management': [0, 0, 1, 0, 0, 0],
            'Marketing': [0, 0, 0, 1, 0, 0],
            'Operations': [0, 0, 0, 0, 1, 0],
            'Web': [0, 0, 0, 0, 0, 1]
        }
        UNIT_VALUES.get(UNIT, [1, 0, 0, 0, 0, 0])  # Defaulting to Finance if not found

        # Combining all features for prediction
        features = np.concatenate([
            standardized_features.flatten(), 
            SEX_VALUES[SEX], 
            DESIGNATION_VALUES[DESIGNATION], 
            UNIT_VALUES[UNIT]
        ])

        prediction = model.predict([features])
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text="The Predicted Salary is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)

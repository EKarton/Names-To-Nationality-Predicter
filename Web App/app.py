import sys
sys.path.append('../ML Model/src')

from flask import Flask, render_template, request, abort
from names_to_nationality_classifier import NamesToNationalityClassifier

def get_countries():
    country_id_to_country_name = {}

    with open('../ML Model/data/countries.csv') as countries_file_reader:

        line = countries_file_reader.readline()
        while line:
            tokenized_line = line.split(',')
            if len(tokenized_line) == 3:
                country_id = int(tokenized_line[0])
                country_name = tokenized_line[1]
                nationality = tokenized_line[2]

                country_id_to_country_name[country_id] = (country_name, nationality)

            line = countries_file_reader.readline()

    return country_id_to_country_name
    
country_id_to_country = get_countries()
countries = [ country_id_to_country[id][0] for id in country_id_to_country ]

# Get the ML model
classifier = NamesToNationalityClassifier(countries)
classifier.load_model_from_file('../ML Model/data/data.npz')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('home/index.html')
  
@app.route('/nationality', methods=['GET', 'POST'])
def get_nationality():
    if request.method == 'POST':
        name = request.form.get('name')
        country = request.form.get('country')
        print('Fixing prediction for name', name, 'with country', country)
        classifier.train_example(name, country)

        return render_template('fix-nationality/index.html')

    else:
        name = request.args.get('name')
        print(name)

        if type(name) is not str:
            abort(404)

        name.strip()

        if len(name) == 0:
            return render_template('prediction-error/index.html', error_message='your name cannot be blank.'), 400

        if len(name.split(' ')) <= 1:
            return render_template('prediction-error/index.html', error_message='your name needs to have a last name.'), 400

        prediction = classifier.predict(name)
        most_probable_country = prediction[0][1]
        return render_template('nationality/index.html', name=name, most_probable_country=most_probable_country, predictions=prediction)            

@app.errorhandler(400)
def not_found_error(error):
    return render_template('400.html'), 400

if __name__ == '__main__':
    app.run(debug = True)
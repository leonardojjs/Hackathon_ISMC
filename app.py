import os
import flask
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
import requests
from OFANI import calculate
import re
from EQTransformer.utils.hdf5_maker import preprocessor
from EQTransformer.core.predictor import predictor



app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER_OFANI = os.path.join(BASE_DIR, 'UPLOAD_OFANI')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static/RESULTS_OFANI')
# Add the UPLOAD_FOLDER to the Flask configuration
app.config['UPLOAD_FOLDER_OFANI'] = UPLOAD_FOLDER_OFANI
UPLOAD_FOLDER_PHI314 = os.path.join(BASE_DIR, 'UPLOAD_PHI314')
RESULTS_FOLDER_PHI314 = os.path.join(BASE_DIR, 'static/RESULTS_PHI314')
app.config['UPLOAD_FOLDER_PHI314'] = UPLOAD_FOLDER_PHI314
json_basepath = os.path.join(os.getcwd(),"json/station_list.json")
ALLOWED_EXT_OFANI = set(['txt', 'csv', 'xlsx'])
ALLOWED_EXT_PHI314 = set(['mseed'])

def allowed_file_OFANI(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT_OFANI

def allowed_file_PHI314(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT_PHI314

def is_matching_pattern(filename):
    # Define a regular expression pattern for matching the filename pattern
    pattern = r'^PB\.B921\..+__\d{8}T\d{6}Z__\d{8}T\d{6}Z\.mseed$'
    return bool(re.match(pattern, filename))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/SeisMoT')
def SeisMoT():
    return render_template("data.html")

@app.route('/OFANI')
def OFANI():
    return render_template("OFANI.html")

@app.route('/success', methods=['GET', 'POST'])
def upload_data_OFANI():
    error = ''
    if request.method == 'POST':
        if 'file' not in request.files:
            error = 'No file part'
            return render_template('OFANI.html', error=error)

        file = request.files['file']

        if file.filename == '':
            error = 'No selected file'
            return render_template('OFANI.html', error=error)

        if not allowed_file_OFANI(file.filename):
            error = 'Please upload txt, csv, xlsx extension only'
            return render_template('OFANI.html', error=error)

        # Save the file to a temporary location
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER_OFANI'], filename)
        file.save(filepath)

        # Assuming the calculate function is defined somewhere in your code
        results = calculate(filepath)

        # Extract the base filename (without extension) from the uploaded filename
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        # Construct the img filenames by appending "_Plot{i}.png" to the base filename for i in range(1, 11)
        image_filenames = [f"{base_filename}_Plot{i}.png" for i in range(1, 11)]

        # Unpack the list into individual variables
        img1, img2, img3, img4, img5, img6, img7, img8, img9, img10 = image_filenames

        if results:
            return render_template('success-ofani.html',
                                   img1=img1,
                                   img2=img2,
                                   img3=img3,
                                   img4=img4,
                                   img5=img5,
                                   img6=img6,
                                   img7=img7,
                                   img8=img8,
                                   img9=img9,
                                   img10=img10,
                                   mainshock_magnitude=results['mainshock_magnitude'],
                                   mainshock_time=results['mainshock_time'],
                                   minimum_magnitude=results['minimum_magnitude'],
                                   maximum_radius=results['maximum_radius'],
                                   start_time=results['start_time'],
                                   end_time=results['end_time'],
                                   events_selected=results['events_selected'],
                                   Tmc_mean=results['Tmc_mean'],
                                   Tmc_max=results['Tmc_max'],
                                   Tmc_min=results['Tmc_min'],
                                   Rmin=results['Rmin'],
                                   Rssm=results['Rssm'],
                                   Rseq=results['Rseq']
                                   )

        else:
            return render_template('OFANI.html', error=error)

    return render_template('OFANI.html')

@app.route('/PHI314')
def PHI314():
    return render_template("PHI314.html")

@app.route('/success_PHI314', methods=['GET', 'POST'])
def upload_data_PHI314():
    error = ''
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'files[]' not in request.files:
            error = 'No file part'
            return render_template('PHI314.html', error=error)

        files = request.files.getlist('files[]')

        # Check if there are no selected files
        if len(files) == 0:
            error = 'No selected files'
            return render_template('PHI314.html', error=error)

        success = True  # Initialize success flag

        for file in files:
            # Check file extension
            if not allowed_file_PHI314(file.filename):
                error = 'Please upload MiniSEED extension only'
                success = False
                break

            # Save each file to a temporary location
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER_PHI314'], filename)
            file.save(filepath)

            # Check if the filename matches the pattern
            if not is_matching_pattern(filename):
                error = ''
                success = False
                break

        if success:
            return render_template('success-phi314.html')
        else:
            return render_template('PHI314.html', error=error)

    return render_template('PHI314.html')

@app.route('/helps')
def helps():
    return render_template("helps.html")

if __name__ == "__main__":
    app.run(debug=True)
    app.config['TEMPLATES_AUTO_RELOAD'] = True


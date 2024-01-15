import os
json_basepath = os.path.join(os.getcwd(), "./json/station_list.json")

from EQTransformer.utils.hdf5_maker import preprocessor

preprocessor(preproc_dir="/preproc", mseed_dir='downloads_mseeds', stations_json=json_basepath, overlap=0.3, n_processor=16)

from EQTransformer.core.predictor import predictor

predictor(input_dir= 'downloads_mseeds_processed_hdfs', input_model='./ModelsAndSampleData/EqT_original_model.h5', output_dir='detections', detection_threshold=0.3, P_threshold=0.1, S_threshold=0.1, number_of_plots=100, plot_mode='time')
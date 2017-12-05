# cmpt318_submit

There are three codes show our work: get_dataframe.py, build_model.py, project_analyzation.ipynb.
get_dataframe.py generates a dataframe contains features of images and corresponding weather labels.
build_model.py reads the generated dataframe, builds and trains a model, returns score of the model.
The selection of the model we used in build_model.py is based on our tests for different models and parameters in test.ipynb.
Model with highest score will be selected. project_analyzation.ipynb shows our progress on testing data and training models.
Put project_analyzation.ipynb and image_labeled.csv under the same directory in Jupyter will allow it to run normally.  
1. Required libraries:
	get_dataframe.py requires following libraries: pandas, numpy, cv2, glob, sys, re.
	build_model.py requires following libraries: pandas, sys, sklearn

2. Order of execution: execute get_dataframe.py, then execute build_model.py

Important!!!!!!!!!!!: execution of get_dataframe.py may take about 2 hours to finish.
		      We have provided the output dataframe under the given folder named 'image_labeled.csv'.
		      Please just execute build_model.py with provided dataframe to save your time.
		      Execution of build_model.py takes about 3 and a half minutes, please be patient on execution.		      	

3. Commands:
	Run 'python3 get_dataframe.py yvr-weather katkam-scaled' to execute get_dataframe.py
		Check following before execute: Provided folders 'yvr-weather' and 'katkam-scaled' are in same directory with get_dataframe.py
	Run 'python3 build_model.py image_labeled.csv' to execute build_model.py
		Check following before execute: Provided file 'image_labeled.csv' are in same directory with build_model.py 

4. Files produced/expected:
	get_dataframe.py returns a csv file, 'image_labeled.csv' under the same directory.
	build_model.py will print the score of the chosen model for this problem.

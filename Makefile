# Makefile to automate setup and running the KMeans web application

# Command to install dependencies
install:
	pip install -r requirements.txt

# Command to run the web application on localhost:3000
run:
	FLASK_APP=app.py flask run --host=0.0.0.0 --port=3000

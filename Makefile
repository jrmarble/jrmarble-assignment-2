# Install dependencies for the project
install:
	pip install -r requirements.txt

# Run the application on localhost:3000
run:
	FLASK_APP=app.py flask run --host=0.0.0.0 --port=3000

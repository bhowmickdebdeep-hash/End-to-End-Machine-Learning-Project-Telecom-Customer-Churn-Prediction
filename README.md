# End-to-End-Machine-Learning-Project-Telecom-Customer-Churn-Prediction
In this repository, we have performed the end to end Exploratory Data Analysis, and idenfitied the characteristics of the customers that are more likely to churn, and I have used them wisely to create a model, and lately, have deployed the model.

🟢 For EDA, please refer to : Churn Analysis - EDA.ipynb
🟢 For Model Building, please refer to: Churn Analysis - Model Building.ipynb
🟢 For Model Deployment, please refer to app.py
🔵 Creating the flask API
app = Flask("__name__")
The loadPage method calls our home.html.

@app.route("/")
def loadPage():
	return render_template('home.html', query="")
The predict method is our POST method, which is basically called when we pass all the inputs from our front end and click SUBMIT.

@app.route("/", methods=['POST'])
def predict():
The run() method of Flask class runs the application on the local development server.

app.run()
Yay, our model is ready, let’s test our bot. The above given Python script is executed from Python shell.

Go to Anaconda Prompt, and run the below query.

python app.py
Below message in Python shell is seen, which indicates that our App is now hosted at http://127.0.0.1:5000/ or localhost:5000

* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
HERE'S HOW OUR FRONTEND LOOKS LIKE:
<img width="1920" height="1080" alt="Screenshot (298)" src="https://github.com/user-attachments/assets/6f56f3a8-2b01-4b46-b767-eec285b1b6f4" />


Customer Retention

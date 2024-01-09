import time

from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/results', methods=['POST'])
def results():
    if request.method == 'POST':
        search_query = request.form['search_query']
        time.sleep(3)
        # Perform any processing or fetching results based on the search query
        # For simplicity, let's just pass the search query to the results template
        return render_template('results.html', search_query=search_query)


if __name__ == '__main__':
    app.run(debug=True)

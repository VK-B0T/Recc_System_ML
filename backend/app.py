from flask import Flask, render_template, request
import pandas as pd
from model import recommend_cpus, recommend_gpus, recommend_rams, recommend_disks

app = Flask(__name__)

# Load datasets
cpu_data = pd.read_csv('C:/Users/Varun/Desktop/Minor Project (RandomForest)/Processed Datasets/CPU.csv')
gpu_data = pd.read_csv('C:/Users/Varun/Desktop/Minor Project (RandomForest)/Processed Datasets/GPU.csv')
ram_data = pd.read_csv('C:/Users/Varun/Desktop/Minor Project (RandomForest)/Processed Datasets/RAM.csv')
disk_data = pd.read_csv('C:/Users/Varun/Desktop/Minor Project (RandomForest)/Processed Datasets/DISK.csv')

# Route for the homepage
@app.route('/')
def home():
    return render_template('home.html')

# Route for the form page
@app.route('/form.html')
def form():
    return render_template('form.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    budget = float(request.form['budget'])
    usage = request.form['usage']
    
    # Allocate budget portions to each component
    cpu_budget = budget * 0.30
    gpu_budget = budget * 0.35
    ram_budget = budget * 0.10
    disk_budget = budget * 0.15

    # Get recommendations for each component
    cpu_recs = recommend_cpus(cpu_data, cpu_budget, usage)
    gpu_recs = recommend_gpus(gpu_data, gpu_budget, usage)
    ram_recs = recommend_rams(ram_data, ram_budget, usage)
    disk_recs = recommend_disks(disk_data, disk_budget, usage)
    print(disk_recs)
    return render_template(
        'recommendations.html', 
        cpu_recs=cpu_recs, 
        gpu_recs=gpu_recs, 
        ram_recs=ram_recs, 
        disk_recs=disk_recs
    )

if __name__ == '__main__':
    app.run(debug=True)

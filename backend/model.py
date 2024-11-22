import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# CPU recommendation
def categorize_cpu_usage(row):
    if row['Benchmark'] < 2000:
        return 'Student'
    elif 2000 <= row['Benchmark'] < 5000:
        return 'Work/Home'
    elif 5000 <= row['Benchmark'] < 8000:
        return 'Gaming or High-end'
    else:
        return 'High-End'

def recommend_cpus(cpu_data, budget, usage):
    cpu_data['Usage'] = cpu_data.apply(categorize_cpu_usage, axis=1)
    cpu_data['Power_Category'] = LabelEncoder().fit_transform(cpu_data['Power_Category'])
    
    filtered_cpus = cpu_data[cpu_data['Price'] <= budget]
    target_data = filtered_cpus[filtered_cpus['Usage'] == usage]
    
    if target_data.empty:
        return "No CPUs available within the specified budget and usage."

    features = target_data[['Benchmark', 'Power_Category']]
    target = target_data['Usage']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    target_data['Prediction'] = model.predict(features)
    recommendations = target_data.sort_values(by='Benchmark', ascending=False).head(3)
    
    return recommendations[['CPU_Name', 'Price', 'Benchmark']].to_dict(orient='records')

# GPU recommendation
def categorize_gpu_usage(row):
    if row['VRAM'] <= 4:
        return 'Student'
    elif 4 < row['VRAM'] <= 8:
        return 'Work/Home'
    elif 8 < row['VRAM'] <= 12:
        return 'Gaming or High-end'
    else:
        return 'High-End'

def recommend_gpus(gpu_data, budget, usage):
    filtered_gpus = gpu_data[(gpu_data['Price'] <= budget) & (gpu_data['Usage'] == usage)]
    if filtered_gpus.empty:
        return []
    features = filtered_gpus[['VRAM', 'Clock_Speed']]
    target = filtered_gpus['Usage']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    filtered_gpus['Prediction'] = model.predict(features)
    recommendations = filtered_gpus.sort_values(by='VRAM', ascending=False).head(3)
    return recommendations[['GPU_Name', 'Price', 'memory', 'model', 'clock_speed']].to_dict(orient='records')

# RAM recommendation
def categorize_ram_usage(row):
    if row['Size'] <= 8:
        return 'Student'
    elif 8 < row['Size'] <= 16:
        return 'Work/Home'
    elif 16 < row['Size'] <= 32:
        return 'Gaming or High-end'
    else:
        return 'High-End'

def recommend_rams(ram_data, budget, usage):
    ram_data['Usage'] = ram_data.apply(categorize_ram_usage, axis=1)
    ram_data['Version'] = LabelEncoder().fit_transform(ram_data['Version'])
    
    filtered_rams = ram_data[ram_data['Price'] <= budget]
    target_data = filtered_rams[filtered_rams['Usage'] == usage]
    
    if target_data.empty:
        return "No RAMs available within the specified budget and usage."

    features = target_data[['Speed', 'Version']]
    target = target_data['Usage']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    target_data['Prediction'] = model.predict(features)
    recommendations = target_data.sort_values(by='Speed', ascending=False).head(3)
    
    return recommendations[['RAM_Name', 'Price', 'Speed', 'Size_GB']].to_dict(orient='records')

# Disk recommendation
def categorize_disk_usage(row):
    if row['Capacity (in GBs)'] < 600:
        return 'Student'
    elif 600 <= row['Capacity (in GBs)'] < 1024:
        return 'Work/Home'
    elif 1024 <= row['Capacity (in GBs)'] <= 1500:
        return 'Gaming or High-end'
    else:
        return 'High-End'

def recommend_disks(disk_data, budget, usage):
    disk_data['Usage'] = disk_data.apply(categorize_disk_usage, axis=1)
    disk_data['Type'] = LabelEncoder().fit_transform(disk_data['Type'])
    
    filtered_disks = disk_data[disk_data['Price'] <= budget]
    target_data = filtered_disks[filtered_disks['Usage'] == usage]
    
    if target_data.empty:
        return "No Disk Drives available within the specified budget and usage."

    features = target_data[['Benchmark Rank', 'Type']]
    target = target_data['Usage']
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    target_data['Prediction'] = model.predict(features)
    recommendations = target_data.sort_values(by='Benchmark Rank', ascending=False).head(3)
    print(recommendations[['driveName', 'Price', 'Capacity (in GBs)', 'Type-1']].to_dict(orient='records'))
    return recommendations[['driveName', 'Price', 'Capacity (in GBs)', 'Type-1']].to_dict(orient='records')

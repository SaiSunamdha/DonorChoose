donors-approval-service/



1. ├─ app/
2. │  ├─ app.py                 # Flask API
3. │  └─ model/
4. │     ├─ model.joblib        # serialized pipeline (created by training)
5. │     └─ meta.json           # feature lists & version info
6. ├─ train/
7. │  └─ train_model.py         # trains and saves pipeline
8. ├─ requirements.txt
9. └─ README.md                 # optional

## Create & activate a virtual environment

1. python -m venv .venv
2. .venv\Scripts\Activate.ps1
3. pip install -r requirements.txt



# Run training
1. Put your CSV at train/data.csv 
2. python train/train_model.py

##  Run the API
python -m waitress --port=8000 app.app:app



## check ::  http://127.0.0.1:5000/health

PowerShell (recommended on Windows)
$body = @{
  teacher_prefix = "Mrs."
  school_state = "IN"
  project_grade_category = "Grades PreK-2"
  project_subject_categories = "Literacy & Language"
  teacher_number_of_previously_posted_projects = 0
  essay_length = 120
  total_cost = 43.77
  teacher_experience = 0
  submit_month = 12
  submit_dow = 1
  num_resources = 3
  combined_essays = "My students are English learners and we need DVDs to practice phonics."
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:8000/predict" -Method Post -ContentType "application/json" -Body $body







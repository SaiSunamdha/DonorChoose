donors-approval-service/
├─ app/
│  ├─ app.py                 # Flask API
│  └─ model/
│     ├─ model.joblib        # serialized pipeline (created by training)
│     └─ meta.json           # feature lists & version info
├─ train/
│  └─ train_model.py         # trains and saves pipeline
├─ requirements.txt
└─ README.md                 # optional

## Create & activate a virtual environment

python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt



# Run training
# Put your CSV at train/data.csv 
python train/train_model.py

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







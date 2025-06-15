# Healthcare Disparities Detection and Mitigation System

## Problem Statement

Healthcare disparities are a big deal, and they hit hard in places like urban India, where marginalized communities often get the short end of the stick when it comes to medical care. We’re talking about unequal treatment, worse health outcomes, and sometimes even a 20% higher mortality rate for certain groups—like women, ethnic minorities, or low-income patients. It’s not just a fairness issue; it’s a life-or-death one. The root causes? Systemic biases baked into healthcare practices, inconsistent protocols across hospitals, and no real way to spot these problems as they happen. Without something to catch and fix these inequities in real-time, trust in the system crumbles, costs skyrocket from untreated conditions, and entire communities suffer. Right now, there’s no unified AI tool out there that can step in, detect these gaps, and help healthcare providers act fast—which is a huge missed opportunity for making care more equitable.

## Solution Approach

So, I decided to build something to fix this mess: an AI-powered platform that digs into healthcare data to spot disparities and nudge providers toward fairer care. I started by using supervised learning to analyze synthetic datasets (like Synthea) and pick up patterns—like when certain groups get delayed diagnoses or less effective treatments. To keep things fair, I baked in fairness metrics to check if the model’s decisions are equitable across demographics like gender or ethnicity. Then, I added a generative AI piece that spits out practical suggestions—think staff training sessions or policy tweaks—based on what the system finds.

For the folks using it, I created a real-time dashboard with Streamlit that ties into electronic health record (EHR) systems via APIs, so hospital admins can see what’s going wrong as it happens. Transparency’s huge here, so I used Explainable AI (SHAP) to show why the system flags certain things, building trust with users. Privacy’s covered too—federated learning keeps patient data safe while letting the system scale across multiple hospitals. I leaned on open-source tools like TensorFlow and Fairlearn to keep it affordable and effective. It’s a mix of tech and practicality, aimed at making healthcare a little less unfair, one hospital at a time.

## Project Overview

My project—the **Healthcare Disparities Detection and Mitigation System**—is all about spotting and fixing inequities in healthcare using Python and machine learning. It’s not just a data cruncher; it’s a tool that gives doctors, nurses, and admins clear, actionable insights. Here’s what it brings to the table, with a bit more detail on how each piece works:

- **Bias Detection**: This is the core. A supervised learning model chews through patient data—think treatment records and outcomes—and flags disparities. For example, it might notice women over 50 get fewer follow-ups than men in the same age group. It’s trained on synthetic data but built to handle real-world messiness.

- **Fairness Metrics**: I didn’t want the model to just guess; I wanted proof it’s fair. This feature calculates selection rates—like how often a group gets a certain treatment—and compares them across demographics. If one group’s getting shortchanged, it’s obvious right away.

- **Risk Scoring**: Patients get a risk score based on how likely they are to face disparities. It’s a heads-up for providers to prioritize care where it’s needed most, like flagging a minority patient who’s overdue for a screening.

- **Generative Interventions**: Once a disparity pops up, the system doesn’t just sit there—it suggests fixes. Maybe it’s extra training for staff on cultural sensitivity or a policy change to standardize diagnostics. It’s powered by generative AI, so the ideas are tailored and practical.

- **Explainable AI (SHAP)**: Nobody trusts a black box, right? SHAP breaks down the model’s decisions—like showing how much gender or race factored into a prediction—so users can see the “why” behind the alerts.

- **Real-Time Auditing**: Every action the system suggests or flags gets logged. It’s like a paper trail for accountability, so hospitals can track what they’ve done (or didn’t do) about disparities.

- **Interactive Dashboard**: Built with Streamlit, this is where users live. It’s got charts, risk scores, and intervention ideas, all in real-time. Even non-techy folks—like a busy nurse—can jump in and use it without a headache.

- **Community Feedback**: Patients and staff can report disparities they see, and if complaints pile up, the system sounds an alert. It’s a way to keep the tool grounded in real experiences, not just data.

I designed it to be intuitive and hands-on, so anyone in healthcare can pick it up and make a difference without needing a PhD in AI.

## Folder Structure

Here’s how I’ve got everything organized in the project—keeps it tidy and easy to navigate:

```
healthcare_disparities_project/
├── data/
│   ├── synthea/
│   │   ├── patients.csv
│   │   ├── encounters.csv
│   ├── preprocessed_data.csv
│   └── interventions_kb.csv
├── models/
│   ├── model.pkl
│   └── risk_model.pkl
├── results/
│   ├── fairness_metrics.csv
│   ├── interventions.txt
│   ├── shap_plot.png
│   ├── audit_log.csv
│   ├── feedback.csv
│   └── complaints.csv
├── src/
│   ├── preprocess_data.py
│   ├── bias_detection.py
│   ├── fairness_analysis.py
│   ├── interventions.py
│   ├── shap_analysis.py
│   ├── audit_trail.py
│   ├── risk_scoring.py
│   ├── dashboard.py
│   └── alert.wav
├── requirements.txt
└── README.md
```

- **data/**: Where all the raw and cleaned-up data lives. The `synthea/` folder has synthetic patient info, while `preprocessed_data.csv` is ready for analysis. `interventions_kb.csv` is a knowledge base for suggested fixes.

- **models/**: Home for the trained models—like `model.pkl` for bias detection and `risk_model.pkl` for scoring patients.

- **results/**: Outputs go here. Fairness metrics, intervention logs, SHAP plots, audit trails, and feedback files—everything you’d want to review later.

- **src/**: All the Python scripts that make the magic happen. From preprocessing data to running the dashboard, each file has a job. Oh, and `alert.wav` is a little sound file for when complaints hit a threshold.

- **requirements.txt**: Just a list of packages you’ll need to install to run this thing.

- **README.md**: You’re reading it! A guide to what this project’s all about.

## Setup and Usage

Getting it up and running is pretty straightforward:

1. **Clone the repo**:

   ```bash
   git clone https://github.com/Kavya30S/healthcare-disparities.git
   ```
2. **Install the dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
3. **Run the scripts**:

   ```bash
   python src/preprocess_data.py
   python src/bias_detection.py
   python src/fairness_analysis.py
   python src/interventions.py
   ```
4. **Fire up the dashboard**:

   ```bash
   streamlit run src/dashboard.py

   ```

   5. **Wroking Deployemnet Link**:

 
   ```bash
   https://healthcaredisparities-kavya30s.streamlit.app/

   ```
   1. **Git Repo Link**:

   ```bash
   git clone https://github.com/Kavya30S/healthcare-disparities.git
   ```

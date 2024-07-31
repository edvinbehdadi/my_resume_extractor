from fastapi import FastAPI, HTTPException, UploadFile, File
import pandas as pd
from resume_extractor import ResumeExtractor
from job_description_extractor import JobDescriptionExtractor
from model_trainer import ModelTrainer
from comparison_utils import (
    compare_with_chatgpt_job_title,
    compare_with_chatgpt_education,
    compare_with_chatgpt_location,
    compare_age_range_with_description
)
from synthetic_data import create_synthetic_data
import os

app = FastAPI()

def main(resume_text, job_description):
    openai_api_key = os.getenv('OPENAI_API_KEY')
    ner_model_name_or_path = "NLPclass/Named-entity-recognition"
    skill_model_name_or_path = "GalalEwida/lm-ner-skills-recognition"

    resume_extractor = ResumeExtractor(ner_model_name_or_path, openai_api_key)
    job_description_extractor = JobDescriptionExtractor(openai_api_key)

    full_name, loc, age, skills, education_resume, title_job_resume = resume_extractor.extract_resume_info(resume_text, skill_model_name_or_path)
    job_skills, education_job, title_job, location, age_DS = job_description_extractor.extract_job_info(job_description, skill_model_name_or_path)

    education_match = compare_with_chatgpt_education(education_resume, education_job, openai_api_key)
    title_job_match = compare_with_chatgpt_job_title(title_job_resume, title_job, openai_api_key)
    title_loc_match = compare_with_chatgpt_location(loc, location, openai_api_key)
    title_age_match = compare_age_range_with_description(age, age_DS, openai_api_key)

    synthetic_data = create_synthetic_data(job_skills, education_job, title_job, location, age_DS)
    synthetic_data.to_csv('synthetic_data.csv')
    model_trainer = ModelTrainer(synthetic_data)
    best_model = model_trainer.train_models()

    input_data = {skill: 1 if skill in skills else 0 for skill in job_skills}
    input_data[education_job] = education_match
    input_data[title_job] = title_job_match
    input_data[location] = title_loc_match
    input_data[age_DS] = title_age_match

    input_df = pd.DataFrame([input_data])
    input_df.to_csv('input_df.csv')
    predicted_target = best_model.predict(input_df)

    return {
        "full_name": full_name,
        "location": loc,
        "age": age,
        "age_DS": age_DS,
        "skills": skills,
        "education_resume": education_resume,
        "title_job_resume": title_job_resume,
        "job_skills": job_skills,
        "education_job": education_job,
        "title_job": title_job,
        "location_job": location,
        "predicted_target": predicted_target[0]
    }

@app.post("/extract")
async def extract(resume_file: UploadFile = File(...), job_description_file: UploadFile = File(...)):
    try:
        resume_text = await resume_file.read()
        job_description = await job_description_file.read()
        
        # Convert bytes to string
        resume_text = resume_text.decode('utf-8')
        job_description = job_description.decode('utf-8')
        
        output = main(resume_text, job_description)
        return output
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
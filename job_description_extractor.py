import openai
import openai
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
class JobDescriptionExtractor:
    def __init__(self, openai_api_key):
        openai.api_key = openai_api_key


    def extract_skills(self, text, skill_model_name_or_path):
        skill_tokenizer = AutoTokenizer.from_pretrained(skill_model_name_or_path)
        skill_model = AutoModelForTokenClassification.from_pretrained(skill_model_name_or_path)
        inputs = skill_tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = skill_model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)
        tokens = skill_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        tags = [skill_model.config.id2label[p.item()] for p in predictions[0]]
        skills = []
        temp_skill = ""
        for token, tag in zip(tokens, tags):
            if tag == "B-TECHNOLOGY":
                if temp_skill:
                    skills.append(temp_skill.strip())
                    temp_skill = ""
                skills.append(token)
            elif tag == "B-TECHNICAL":
                if temp_skill:
                    skills.append(temp_skill.strip())
                    temp_skill = ""
                temp_skill = token
            elif tag == "I-TECHNICAL":
                temp_skill += token.replace('##', '')
        if temp_skill:
            skills.append(temp_skill.strip())
        return list(set(skills))

    def translate_text(self, text, target_language="en"):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates text."},
                {"role": "user", "content": f"Translate the following text to {target_language}:\n\n{text}"}
            ],
            max_tokens=1000
        )
        return response.choices[0].message["content"].strip()
    def extract_location(self, job_description):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts information from text."},
                {"role": "user", "content": f"Extract location from the following job description:\n\n{job_description}"}
            ],
            max_tokens=1000
        )
        return response.choices[0].message["content"].strip()



    def title(self, text):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts information from text."},
                {"role": "user", "content": f"Extract the [Last Job Title] from the following text:\n\n{text}"}
            ],
            max_tokens=1000
        )
        return response.choices[0].message["content"].strip()

    def extract_education(self, text):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts information from text."},
                {"role": "user", "content": f"Extract the [Highest Education Degree] from the following text:\n\n{text}"}
            ],
            max_tokens=1000
        )
        return response.choices[0].message["content"].strip()
    def extract_age_range(self, text):
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts information from text."},
            {"role": "user", "content": f"Extract the age range from the following text:\n\n{text}"}
        ],
        max_tokens=1000
    )
        return response.choices[0].message["content"].strip()

        pass

    def extract_job_info(self, job_description, skill_model_name_or_path):
        # تابع استخراج اطلاعات کلی از توصیف شغلی
        translated_job_description = self.translate_text(job_description)
        job_skills = self.extract_skills(translated_job_description, skill_model_name_or_path)
        education_job = self.extract_education(translated_job_description)
        title_job = self.title(translated_job_description)
        location = self.extract_location(translated_job_description)
        age_DS = self.extract_age_range(translated_job_description)
        return job_skills, education_job, title_job, location, age_DS
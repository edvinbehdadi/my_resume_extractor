
import openai
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import torch
class ResumeExtractor:
    def __init__(self, ner_model_name_or_path, openai_api_key):
        self.ner_model_name_or_path = ner_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(ner_model_name_or_path)
        self.model = AutoModelForTokenClassification.from_pretrained(ner_model_name_or_path)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
        openai.api_key = openai_api_key

    def calculate_age(self, date_string):
        current_year = 1403
        ymd_match = re.match(r'(\d{1,4})/(\d{1,2})/(\d{1,2})', date_string)
        if ymd_match:
            year = int(ymd_match.group(1))
            if len(ymd_match.group(1)) == 4:
                age = current_year - year
            else:
                year += 1300
                age = current_year - year
            return age
        four_digit_match = re.match(r'(13\d{2})', date_string)
        if four_digit_match:
            year = int(four_digit_match.group(1))
            age = current_year - year
            return age
        return None

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
    def extract_ner_info(self, text):
        ner_results = self.nlp(text)
        full_name = ''
        loc = ''
        age = None
        i = 0
        while i < len(ner_results):
            if ner_results[i]['entity'] == 'B-pers' and ner_results[i]['score'] >= 0.80:
                if full_name:
                    full_name += ' '
                full_name += ner_results[i]['word']
                current_score = ner_results[i]['score']
                stop_adding = False
                for j in range(i + 1, len(ner_results)):
                    if ner_results[j]['entity'] == 'I-pers' and ner_results[j]['score'] >= 0.80:
                        if ner_results[j]['score'] >= current_score * 0.90:
                            full_name += ner_results[j]['word'].replace('##', '')
                            current_score = ner_results[j]['score']
                            i = j
                        else:
                            stop_adding = True
                            break
                    else:
                        stop_adding = True
                        break
                if stop_adding:
                    break
            i += 1
        for entity in ner_results:
            if entity['entity'] in ['B-loc', 'I-loc']:
                if loc:
                    loc += ' '
                loc += entity['word']
        age_match = re.search(r'سن\s*:\s*(\d+)', text)
        if age_match:
            age = int(age_match.group(1))
        else:
            date_match = re.search(r'(\d{1,4}/\d{1,2}/\d{1,2})', text)
            if date_match:
                age = self.calculate_age(date_match.group(1))
            else:
                four_digit_match = re.search(r'(13\d{2})', text)
                if four_digit_match:
                    age = self.calculate_age(four_digit_match.group(1))
        return full_name, loc, age

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


    def extract_education_resume(self, text):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts information from text."},
                {"role": "user", "content": f"Extract only the highest education degree and field from the following text:\n\n{text}\n\nFormat the response as 'Degree in Field' and nothing else."}
            ],
            max_tokens=1000
        )
        return response.choices[0].message["content"].strip()

    def extract_job_resume(self, text):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts information from text."},
                {"role": "user", "content": f"Extract only the last job title from the following text:\n\n{text}\n\nProvide just the job title and nothing else."}
            ],
            max_tokens=1000
        )
        return response.choices[0].message["content"].strip()

    def extract_resume_info(self, resume_text, skill_model_name_or_path):
        # تابع استخراج اطلاعات کلی از رزومه
        full_name, loc, age = self.extract_ner_info(resume_text)
        translated_resume = self.translate_text(resume_text)
        skills = self.extract_skills(translated_resume, skill_model_name_or_path)
        education_resume = self.extract_education_resume(translated_resume)
        title_job_resume = self.extract_job_resume(translated_resume)
        return full_name, loc, age, skills, education_resume, title_job_resume
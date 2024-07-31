import openai
import re
def compare_with_chatgpt_job_title(text1, text2, openai_api_key):
    openai.api_key = openai_api_key
    prompt = f"Compare the following two texts and determine if they match in job title . Return 1 for match and 0 for no match.\n\nText 1: {text1}\n\nText 2: {text2}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that helps compare texts for matching job titles "},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )

    # Extract the response content
    result = response.choices[0].message['content'].strip()

    # Check if the response contains '1' or '0' and return the corresponding integer
    if '1' in result:
        return 1
    elif '0' in result:
        return 0
    else:
        raise ValueError(f"Unexpected response: {result}")




def compare_with_chatgpt_education(text1, text2, openai_api_key):
    openai.api_key = openai_api_key
    prompt = f"Compare the following two texts and determine if they match in education . Return 1 for match and 0 for no match.\n\nText 1: {text1}\n\nText 2: {text2}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that helps compare texts for matching education "},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )

    # Extract the response content
    result = response.choices[0].message['content'].strip()

    # Check if the response contains '1' or '0' and return the corresponding integer
    if '1' in result:
        return 1
    elif '0' in result:
        return 0
    else:
        raise ValueError(f"Unexpected response: {result}")


def compare_with_chatgpt_location(text1, text2, openai_api_key):
    openai.api_key = openai_api_key
    prompt = f"Compare the following two texts and determine if they match in location . Return 1 for match and 0 for no match.\n\nText 1: {text1}\n\nText 2: {text2}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that helps compare texts for matching location "},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )

    # Extract the response content
    result = response.choices[0].message['content'].strip()

    # Check if the response contains '1' or '0' and return the corresponding integer
    if '1' in result:
        return 1
    elif '0' in result:
        return 0
    else:
        raise ValueError(f"Unexpected response: {result}")


def compare_age_range_with_description(age, age_DS, openai_api_key):
    openai.api_key = openai_api_key
    
    prompt = (f"Check if the age {age} falls within the age range '{age_DS}' "
              f"Return '1' if it falls within the range, otherwise return '0'.\n\n"
              f"Age: {age}\n\n"
              f"Age Range: {age_DS}")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are an assistant that helps compare age ranges with a given age."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )

    result = response.choices[0].message['content'].strip()

    # استفاده از regex برای پیدا کردن '1' یا '0'
    match = re.search(r"\b[01]\b", result)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"Unexpected response: {result}")      

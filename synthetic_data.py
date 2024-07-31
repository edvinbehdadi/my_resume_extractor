import numpy as np
import pandas as pd

def create_synthetic_data(job_skills, education, title_job, location, age_DS, num_rows=2000):
    if isinstance(job_skills, str):
        job_skills = [job_skills]
    if isinstance(education, str):
        education = [education]
    if isinstance(title_job, str):
        title_job = [title_job]
    if isinstance(location, str):
        location = [location]
    if isinstance(age_DS, str):
        age_DS = [age_DS]

    features = job_skills + education + title_job + location + age_DS
    data = np.random.randint(2, size=(num_rows, len(features)))
    df = pd.DataFrame(data, columns=features)
    df['initial_TARGET'] = df.sum(axis=1)

    min_target = df['initial_TARGET'].min()
    max_target = df['initial_TARGET'].max()
    df['TARGET'] = (df['initial_TARGET'] - min_target) * (100 / (max_target - min_target))
    df.drop(columns=['initial_TARGET'], inplace=True)

    df.loc[df.sum(axis=1) == 0, 'TARGET'] = 0
    df.loc[df.sum(axis=1) == len(features), 'TARGET'] = 100

    return df
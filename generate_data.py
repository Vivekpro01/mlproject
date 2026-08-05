import numpy as np
import pandas as pd

np.random.seed(42)

n = 1000

# gender: ~518 female, ~482 male
gender_choices = ['female'] * 518 + ['male'] * 482
np.random.shuffle(gender_choices)
gender = gender_choices

# race/ethnicity: group A (~89), group B (~190), group C (~319), group D (~262), group E (~140)
race_choices = (
    ['group A'] * 89 +
    ['group B'] * 190 +
    ['group C'] * 319 +
    ['group D'] * 262 +
    ['group E'] * 140
)
np.random.shuffle(race_choices)
race = race_choices

# parental level of education
edu_options = [
    "bachelor's degree", "some college", "master's degree",
    "associate's degree", "high school", "some high school"
]
edu_weights = [0.118, 0.226, 0.059, 0.222, 0.196, 0.179]
parental_edu = np.random.choice(edu_options, size=n, p=edu_weights)

# lunch
lunch = np.random.choice(['standard', 'free/reduced'], size=n, p=[0.645, 0.355])

# test preparation course
test_prep = np.random.choice(['none', 'completed'], size=n, p=[0.642, 0.358])

# math score: integers 0-100, mean ~66
math_score = np.clip(np.round(np.random.normal(loc=66, scale=15, size=n)).astype(int), 0, 100)

# reading score: integers 17-100, mean ~69
reading_score = np.clip(np.round(np.random.normal(loc=69, scale=14, size=n)).astype(int), 17, 100)

# writing score: integers 10-100, mean ~68
writing_score = np.clip(np.round(np.random.normal(loc=68, scale=15, size=n)).astype(int), 10, 100)

df = pd.DataFrame({
    'gender': gender,
    'race/ethnicity': race,
    'parental level of education': parental_edu,
    'lunch': lunch,
    'test preparation course': test_prep,
    'math score': math_score,
    'reading score': reading_score,
    'writing score': writing_score
})

output_path = r'd:\python new programs\mlproject\notebook\data\stud.csv'
df.to_csv(output_path, index=False)
print(f"Saved {len(df)} rows to {output_path}")
print(f"Columns: {list(df.columns)}")
print(f"\nValue counts:")
print(f"  gender:\n{df['gender'].value_counts().to_string()}")
print(f"  race/ethnicity:\n{df['race/ethnicity'].value_counts().sort_index().to_string()}")
print(f"\nScore means:")
print(f"  math score mean: {df['math score'].mean():.2f}")
print(f"  reading score mean: {df['reading score'].mean():.2f}")
print(f"  writing score mean: {df['writing score'].mean():.2f}")

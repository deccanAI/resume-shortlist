# Resume Shortlist API

Resume screening system using GPT models to match candidates with job requirements.

## Features

- **Multi-role**: ML Engineer, MLOps, NLP, CV engineering positions
- **Auto-matching**: Finds best job fit based on skills
- **Experience levels**: Junior/mid/senior classifications
- **Batch processing**: Handles CSV resume data
- **Ratings**: 1-5 scores with explanations

## Flow

1. Load resumes from CSV
2. Extract key info
3. Match to job type (optional)
4. Evaluate using GPT-4o-mini
5. Export results (CSV/JSON)

## Usage
`python3 api_call.py`

result
```
ID: c0b30b4a-1054-4931-9dad-4642948b05eb | Job: ml_engineer | Rating: 4/5 | YoE: 0.915 (jr)
Feedback: The candidate has relevant experience in machine learning and web development, particularly with Python and Flask, which aligns well with the job requirements. They have hands-on experience in developing models (e.g., financial credit models) and have exposure to data handling with SQL. However, they lack experience with some specific tools and frameworks mentioned in the job description, such as TensorFlow or PyTorch, and have limited experience with MLOps practices and CI/CD pipelines. Overall, they are a strong candidate with room for growth in specific areas.
```
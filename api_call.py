import os
import json
import openai
import pandas as pd
import logging
from dotenv import load_dotenv
from job_desc import ml_engineer, mlops, nlp_engineer, cv_engineer, classify_exp

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Job description dictionary for global access
JOB_DESC_DICT = {
    "ml_engineer": ml_engineer,
    "mlops": mlops,
    "nlp_engineer": nlp_engineer,
    "cv_engineer": cv_engineer
}

def determine_job_type(resume_content):
    """Determine the most suitable job type based on the resume content."""
    prompt = f"""As a technical recruiter, determine the best job fit for this candidate:
    
    CANDIDATE RESUME:
    {resume_content}
    
    Return only one of these exact strings: "ml_engineer", "mlops", "nlp_engineer", or "cv_engineer".
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a technical recruiter evaluating candidates."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=20
        )
        
        job_type = response.choices[0].message.content.strip().lower()
        # Return valid job type or default
        return job_type if job_type in JOB_DESC_DICT else "ml_engineer"
    
    except Exception as e:
        logger.error(f"Error determining job type: {str(e)}")
        return "ml_engineer"

def rate_resume(resume_content, job_type="ml_engineer", job_description=None, yoe=None, auto_determine_job=True):
    """Rate a resume against job requirements using GPT-4o-mini."""
    # Auto-determine job type if requested
    if auto_determine_job:
        job_type = determine_job_type(resume_content)
    
    # Get appropriate job description if not provided
    if job_description is None and yoe is not None:
        exp_level = classify_exp(yoe)
        job_description = JOB_DESC_DICT.get(job_type, {}).get(exp_level, ml_engineer["jr"])
    
    prompt = f"""Rate this candidate (1-5) (1 being not qualified, 5 being extremely qualified) against the job requirements:
    
    JOB DESCRIPTION:
    {job_description}
    
    CANDIDATE RESUME:
    {resume_content}
    
    Return JSON format:
    {{
        "rating": <1-5>,
        "explanation": "<brief explanation of strengths and weaknesses>"
    }}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a technical recruiter evaluating candidates."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        result = json.loads(response.choices[0].message.content)
        result["selected_job_type"] = job_type
        return result
    
    except Exception as e:
        logger.error(f"Error rating resume: {str(e)}")
        return {"error": str(e), "rating": 0, "explanation": "Failed to evaluate resume", "selected_job_type": job_type}

def extract_resume_content(resume_data):
    """Extract and format resume content from JSON data"""
    try:
        # Calculate experience
        fte_exp_years = float(resume_data.get('fte_exp_years', 0))
        intern_exp_years = float(resume_data.get('intern_exp_years', 0)) * 0.5
        total_yoe = fte_exp_years + intern_exp_years
        
        # Format resume content more concisely
        resume_content = f"""
        Education:
        - Undergrad: {resume_data.get('undergrad_university', 'N/A')} ({resume_data.get('undergrad_grad_year', 'N/A')})
        - Masters: {resume_data.get('masters_university', 'N/A')} ({resume_data.get('masters_grad_year', 'N/A')})
        Experience: {total_yoe} years total ({fte_exp_years} full-time, {intern_exp_years*2} internship)
        Industries: {', '.join(resume_data.get('workex_industries', ['None']))}
        Companies: {', '.join(resume_data.get('workex_companies', ['None']))}                
        Skills: {', '.join(set().union(
            resume_data.get('fte_skills', []), 
            resume_data.get('intern_skills', []), 
            resume_data.get('project_skills', [])
        ))}              
        Summary: {resume_data.get('summary', 'Not available')}
        """
        
        return resume_content, total_yoe
    except Exception as e:
        logger.error(f"Error extracting resume content: {str(e)}")
        return "Error extracting resume data", 0

def process_resume_from_csv(csv_file_path, job_type="ml_engineer", auto_determine_job=True):
    """Process resumes from CSV file and rate them."""
    try:
        if not os.path.exists(csv_file_path):
            logger.error(f"CSV file not found: {csv_file_path}")
            return {"error": f"File not found: {csv_file_path}"}
            
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        logger.info(f"Processing {len(df)} resumes from {csv_file_path}")
        
        results = {}
        
        for index, row in df.iterrows():
            soul_id = row.get('soul_id', f"candidate_{index}")
            
            try:
                # Parse the resume JSON and extract content
                resume_data = json.loads(row.get('gpt_resume', '{}'))
                resume_content, total_yoe = extract_resume_content(resume_data)
                
                # Rate this resume
                result = rate_resume(
                    resume_content, 
                    job_type=job_type, 
                    yoe=total_yoe,
                    auto_determine_job=auto_determine_job
                )
                
                result["years_of_experience"] = total_yoe
                result["experience_level"] = classify_exp(total_yoe)
                results[soul_id] = result
                
                logger.info(f"Processed {soul_id}: Rating {result.get('rating', 0)}/5")
            
            except json.JSONDecodeError:
                results[soul_id] = {"rating": 0, "explanation": "Failed to parse resume JSON"}
                logger.warning(f"Failed to parse resume JSON for {soul_id}")
            except Exception as e:
                results[soul_id] = {"error": str(e), "rating": 0, "explanation": "Failed to process resume"}
                logger.error(f"Error processing {soul_id}: {str(e)}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        return {"error": str(e)}

def save_results_to_csv(results, output_path="resume_ratings.csv"):
    """Save results to both JSON and CSV format"""
    try:
        # Save JSON
        with open('resume_ratings.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create DataFrame and save CSV (more concise with list comprehension)
        rows = [{
            "candidate_id": soul_id,
            "rating": result.get("rating", 0),
            "job_type": result.get("selected_job_type", "N/A"),
            "years_of_experience": result.get("years_of_experience", 0),
            "experience_level": result.get("experience_level", "N/A"),
            "explanation": result.get("explanation", "N/A")
        } for soul_id, result in results.items()]
        
        pd.DataFrame(rows).to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path} and resume_ratings.json")
        return True
    
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        return False

if __name__ == "__main__":
    csv_path = "./Rough - Sheet3.csv"
    AUTO_DETERMINE_JOB = True
    DEFAULT_JOB_TYPE = "ml_engineer"
    
    logger.info(f"Starting resume processing with auto_determine_job={AUTO_DETERMINE_JOB}")
    results = process_resume_from_csv(csv_path, DEFAULT_JOB_TYPE, AUTO_DETERMINE_JOB)
    
    # Print results sorted by rating (more concisely)
    for soul_id, result in sorted(results.items(), key=lambda x: x[1].get("rating", 0), reverse=True):
        print(f"ID: {soul_id} | Job: {result.get('selected_job_type', 'N/A')} | " 
              f"Rating: {result.get('rating', 'N/A')}/5 | " 
              f"YoE: {result.get('years_of_experience', 'N/A')} ({result.get('experience_level', 'N/A')})")
        print(f"Feedback: {result.get('explanation', 'N/A')}")
        print("-" * 40)
    
    save_results_to_csv(results)